#!/usr/bin/env python3
"""Measure Von's serving latency on a real EC2 host, JevBench-style.

Boots one instance (CPU or GPU), installs von-sdk from the S3 source tarball,
syncs checkpoints/von-1.2 from S3, starts `von serve`, runs
benchmarks/measure_latency.py serially against it on the same host, uploads
the JSON to S3 and shuts the instance down. Total cost is bounded by the
hard shutdown at +45 minutes.

    uv run python benchmarks/launch_speed_measure.py --mode cpu
    uv run python benchmarks/launch_speed_measure.py --mode gpu

Fetch results afterwards with:
    aws s3 sync s3://model-weight/speed-measure/ results/speed/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

VENV_AWS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "aws"))
AWS_CLI = VENV_AWS if os.path.exists(VENV_AWS) else (shutil.which("aws") or "aws")
REGION = "us-west-2"
SUBNETS = [
    ("subnet-083ba040", "us-west-2a"),
    ("subnet-070e7461", "us-west-2b"),
    ("subnet-b75369ec", "us-west-2c"),
    ("subnet-a663228e", "us-west-2d"),
]
IAM_PROFILE = "AmazonSSMRoleForInstancesQuickSetup"
S3_RESULTS = "s3://model-weight/speed-measure"
S3_CKPT = "s3://model-weight/von-continue-init-ckpt"  # == checkpoints/von-1.2
S3_SRC = "s3://model-weight/von-marker-src.tar.gz"
S3_PUBLIC = "s3://model-weight/jevbench-public"

MODES = {
    # AMI, instance candidates, device flag, endpoint_kind, extra pip
    "cpu": {
        # Ubuntu 22.04 LTS us-west-2 (Canonical). c7i = Sapphire Rapids, 4 vCPU
        # to sit near JevBench's own "4 threads" CPU condition.
        "ami": "ami-0fbdee0e602a53461",  # resolved via SSM canonical/ubuntu/server/22.04 on 2026-09-27
        "types": ["c7i.xlarge", "c6i.xlarge", "m7i.xlarge"],
        "device": "openvino:cpu",
        "kind": "cpu",
        "pip": "torch --index-url https://download.pytorch.org/whl/cpu",
        "hardware": "AWS c7i.xlarge (4 vCPU Intel Sapphire Rapids), openvino:cpu",
    },
    "gpu": {
        "ami": "ami-0e24e0019a12c5b13",  # DL Base AMI, CUDA, Ubuntu 22.04
        "types": ["g5.xlarge", "g4dn.xlarge", "g6.xlarge"],
        "device": "cuda",
        "kind": "gpu",
        "pip": "torch",
        "hardware": "AWS g5.xlarge (1x NVIDIA A10G), cuda",
    },
}

USER_DATA = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
cleanup() {{
  rc=$?
  echo "=== [EXIT rc=$rc] uploading log and shutting down ==="
  aws s3 cp /var/log/user-data.log {s3_results}/{tag}.log || true
  aws s3 cp /opt/von/serve.log {s3_results}/{tag}.serve.log || true
  shutdown -h now
}}
trap cleanup EXIT
shutdown -c 2>/dev/null || true
shutdown -h +{failsafe_min} &

export DEBIAN_FRONTEND=noninteractive
systemctl stop unattended-upgrades.service 2>/dev/null || true
for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
apt-get -o DPkg::Lock::Timeout=600 -y update
apt-get -o DPkg::Lock::Timeout=600 -y install curl unzip
if ! command -v aws >/dev/null; then
  curl -sS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
fi
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /opt/von && cd /opt/von
aws s3 cp {s3_src} /tmp/src.tar.gz && tar -xzf /tmp/src.tar.gz -C /opt/von
aws s3 sync {s3_ckpt}/ /opt/von/checkpoints/von-1.2/
aws s3 sync {s3_public}/ /opt/von/public/

uv venv --python 3.12 /opt/von/.venv
uv pip install --python /opt/von/.venv {pip}
uv pip install --python /opt/von/.venv transformers accelerate fastapi uvicorn pydantic click httpx openvino numpy safetensors
export PYTHONPATH=/opt/von/src
export VON_DEVICE={device}
export HOME=/root
echo "=== hardware ==="; lscpu | grep -E 'Model name|^CPU\\(s\\)'; nvidia-smi -L 2>/dev/null || true; free -g; df -h / | tail -1

{job}
echo "=== DONE ==="
"""


def aws(cmd):
    r = subprocess.run([AWS_CLI] + cmd + ["--region", REGION, "--output", "json"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


JOBS = {
    "latency": """/opt/von/.venv/bin/python -m von.cli serve --host 127.0.0.1 --port 8000 --device {device} > /opt/von/serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/health || (cat /opt/von/serve.log; exit 1)

/opt/von/.venv/bin/python benchmarks/measure_latency.py --url http://127.0.0.1:8000 \\
  --public /opt/von/public --out /opt/von/latency_{tag}.json \\
  --hardware "{hardware}" --device {device} --endpoint-kind {kind} --tiers standard,hard --warmup 3
aws s3 cp /opt/von/latency_{tag}.json {s3_results}/latency_{tag}.json
aws s3 cp /opt/von/serve.log {s3_results}/{tag}.serve.log || true
""",
    # chain-of-options gate on the 114-item numeric slice; plain baseline reused from the tarball
    # chains on vs off, same checkpoint: jev standard + jabr v2 + jev hard + jev easy (out-of-sample guard)
    "gate_chains": """export JEVBENCH_PUBLIC=/opt/von/public
/opt/von/.venv/bin/python benchmarks/gate_standard.py --device {device} --suites jev_standard,jabr_v2,jev_hard,jev_easy \\
  --cand-chains /opt/von/src/von/chains/library --out /opt/von/gate_{tag}.json 2>&1 | tee /opt/von/serve.log
aws s3 cp /opt/von/gate_{tag}.json {s3_results}/gate_{tag}.json
aws s3 cp benchmarks/data/gate_cache/von-1.2+chains.json {s3_results}/gate_{tag}.cand_cache.json
aws s3 cp benchmarks/data/gate_cache/von-1.2.json {s3_results}/gate_{tag}.base_cache.json
""",
    # same, with the 512-token state cap on the candidate (base is served from cache)
    "gate_chains_cap512": """export JEVBENCH_PUBLIC=/opt/von/public
export VON_MAX_STATE_TOKENS=512
/opt/von/.venv/bin/python benchmarks/gate_standard.py --device {device} --suites jev_standard,jabr_v2,jev_hard,jev_easy \\
  --cand-chains /opt/von/src/von/chains/library --out /opt/von/gate_{tag}.json 2>&1 | tee /opt/von/serve.log
aws s3 cp /opt/von/gate_{tag}.json {s3_results}/gate_{tag}.json
aws s3 cp benchmarks/data/gate_cache/von-1.2+chains.json {s3_results}/gate_{tag}.cand_cache.json
""",
    # Decision Index (multimodalart/jev-decision-index): rebuild the 0.2.1 suite from
    # sources, serve Von with overflow refusal (their no-truncation rule), run their own
    # http engine + scorer over all ~150k requests, upload run dir to S3 and to an HF
    # dataset. HF token from s3://model-weight/secrets/hf_token (HLE is gated).
    "decision_index": """export HF_TOKEN=$(aws s3 cp s3://model-weight/secrets/hf_token - | tr -d '[:space:]')
export HF_HUB_DISABLE_XET=1
[ -n "$HF_TOKEN" ] || (echo "no hf token" && exit 1)
apt-get -o DPkg::Lock::Timeout=600 -y install git
git clone -q https://github.com/apolinario/decision-index /opt/di && cd /opt/di && git checkout -q {di_commit}
uv pip install --python /opt/von/.venv -e "/opt/di[rebuild]" httpx huggingface_hub
cd /opt/di
/opt/von/.venv/bin/python -m decision_index suite rebuild --work /opt/di/work 2>&1 | tail -40
/opt/von/.venv/bin/python -m decision_index suite import \\
  --rows /opt/di/work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \\
  --added-rows /opt/di/work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
aws s3 sync /opt/di/suite-0.2 {s3_results}/di_suite/ --quiet || true

export VON_ON_OVERFLOW=refuse
/opt/von/.venv/bin/python -m von.cli serve --host 127.0.0.1 --port 8000 --device {device} > /opt/von/serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/health || (cat /opt/von/serve.log; exit 1)

# smoke: 200 rows, must score without runtime errors before the full run
/opt/von/.venv/bin/python -m decision_index suite sample --n 200 --out /opt/di/sample-200.jsonl.gz
/opt/von/.venv/bin/python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=von-1.3 \\
  --rows /opt/di/sample-200.jsonl.gz --out /opt/di/runs/smoke 2>&1 | tail -5
/opt/von/.venv/bin/python -m decision_index score --results /opt/di/runs/smoke/results.jsonl 2>&1 | tail -20
aws s3 sync /opt/di/runs/smoke {s3_results}/di_{tag}/smoke/ --quiet

# periodic checkpoint upload so a failsafe shutdown loses nothing (resumable)
( while true; do sleep 900; aws s3 sync /opt/di/runs/von-1.3 {s3_results}/di_{tag}/von-1.3/ --quiet; done ) &
/opt/von/.venv/bin/python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=von-1.3 \\
  --out /opt/di/runs/von-1.3 --upload wfzyx/decision-index-results --upload-path runs/von-1.3 2>&1 | tail -60
aws s3 sync /opt/di/runs/von-1.3 {s3_results}/di_{tag}/von-1.3/ --quiet
aws s3 cp /opt/von/serve.log {s3_results}/{tag}.serve.log || true
""",
    "chains": """export JEVBENCH_PUBLIC=/opt/von/public
/opt/von/.venv/bin/python benchmarks/probe_chains.py --mode bindall --device {device} \\
  --baseline benchmarks/data/chains_gate.json --out /opt/von/chains_{tag}.json 2>&1 | tee /opt/von/serve.log
aws s3 cp /opt/von/chains_{tag}.json {s3_results}/chains_{tag}.json
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--job", choices=JOBS, default="latency")
    ap.add_argument("--failsafe-min", type=int, default=45)
    ap.add_argument("--di-commit", default="87d4650", help="apolinario/decision-index commit for --job decision_index")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--public", default=os.path.expanduser("~/scratch/jevbench/datasets/public"))
    ap.add_argument("--skip-upload", action="store_true")
    a = ap.parse_args()
    m = MODES[a.mode]
    tag = f"{a.job}_{a.mode}_{time.strftime('%Y%m%d_%H%M')}"

    if not a.skip_upload:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Same tarball the training launcher consumes: keep training/ in it.
        subprocess.run(["tar", "-czf", "/tmp/von-src.tar.gz", "-C", root, "src", "benchmarks", "training"], check=True)
        subprocess.run([AWS_CLI, "s3", "cp", "/tmp/von-src.tar.gz", S3_SRC, "--region", REGION], check=True)
        subprocess.run([AWS_CLI, "s3", "sync", a.public, S3_PUBLIC + "/", "--region", REGION,
                        "--exclude", "*", "--include", "*.jsonl"], check=True)

    ud = "/tmp/user_data_speed.sh"
    with open(ud, "w") as f:
        fmt = dict(s3_results=S3_RESULTS, s3_src=S3_SRC, s3_ckpt=S3_CKPT, s3_public=S3_PUBLIC,
                   tag=tag, pip=m["pip"], device=m["device"], kind=m["kind"], hardware=m["hardware"], failsafe_min=a.failsafe_min,
                   di_commit=a.di_commit)
        f.write(USER_DATA.format(job=JOBS[a.job].format(**fmt), **fmt))

    iid = None
    for itype in m["types"]:
        for subnet, az in SUBNETS:
            try:
                res = aws(["ec2", "run-instances", "--image-id", m["ami"], "--instance-type", itype,
                           "--subnet-id", subnet, "--iam-instance-profile", f"Name={IAM_PROFILE}",
                           "--user-data", f"file://{ud}", "--count", "1",
                           "--instance-initiated-shutdown-behavior", "terminate",
                           "--block-device-mappings", json.dumps([{"DeviceName": "/dev/sda1",
                                                                   "Ebs": {"VolumeSize": a.disk_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}]),
                           "--tag-specifications", json.dumps([{"ResourceType": "instance",
                                                                 "Tags": [{"Key": "Name", "Value": f"von-speed-{a.mode}"}]}])])
                iid = res["Instances"][0]["InstanceId"]
                print(f"launched {itype} in {az}: {iid}  tag={tag}")
                break
            except Exception as e:
                s = str(e)
                if any(k in s for k in ("InsufficientInstanceCapacity", "Unsupported", "Unsupported")):
                    continue
                print(f"  {itype}/{az}: {s[:200]}")
        if iid:
            break
    if not iid:
        sys.exit("no capacity")
    print(f"results -> {S3_RESULTS}/latency_{tag}.json ; log -> {S3_RESULTS}/{tag}.log")
    print(json.dumps({"instance_id": iid, "tag": tag, "mode": a.mode}))


if __name__ == "__main__":
    main()
