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
shutdown -h +45 &

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

/opt/von/.venv/bin/python -m von.cli serve --host 127.0.0.1 --port 8000 --device {device} > /opt/von/serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/health || (cat /opt/von/serve.log; exit 1)

/opt/von/.venv/bin/python benchmarks/measure_latency.py --url http://127.0.0.1:8000 \\
  --public /opt/von/public --out /opt/von/latency_{tag}.json \\
  --hardware "{hardware}" --device {device} --endpoint-kind {kind} --tiers standard,hard --warmup 3
aws s3 cp /opt/von/latency_{tag}.json {s3_results}/latency_{tag}.json
aws s3 cp /opt/von/serve.log {s3_results}/{tag}.serve.log || true
echo "=== DONE ==="
"""


def aws(cmd):
    r = subprocess.run([AWS_CLI] + cmd + ["--region", REGION, "--output", "json"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--public", default=os.path.expanduser("~/scratch/jevbench/datasets/public"))
    ap.add_argument("--skip-upload", action="store_true")
    a = ap.parse_args()
    m = MODES[a.mode]
    tag = f"{a.mode}_{time.strftime('%Y%m%d_%H%M')}"

    if not a.skip_upload:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Same tarball the training launcher consumes: keep training/ in it.
        subprocess.run(["tar", "-czf", "/tmp/von-src.tar.gz", "-C", root, "src", "benchmarks", "training"], check=True)
        subprocess.run([AWS_CLI, "s3", "cp", "/tmp/von-src.tar.gz", S3_SRC, "--region", REGION], check=True)
        subprocess.run([AWS_CLI, "s3", "sync", a.public, S3_PUBLIC + "/", "--region", REGION,
                        "--exclude", "*", "--include", "*.jsonl"], check=True)

    ud = "/tmp/user_data_speed.sh"
    with open(ud, "w") as f:
        f.write(USER_DATA.format(s3_results=S3_RESULTS, s3_src=S3_SRC, s3_ckpt=S3_CKPT, s3_public=S3_PUBLIC,
                                 tag=tag, pip=m["pip"], device=m["device"], kind=m["kind"], hardware=m["hardware"]))

    iid = None
    for itype in m["types"]:
        for subnet, az in SUBNETS:
            try:
                res = aws(["ec2", "run-instances", "--image-id", m["ami"], "--instance-type", itype,
                           "--subnet-id", subnet, "--iam-instance-profile", f"Name={IAM_PROFILE}",
                           "--user-data", f"file://{ud}", "--count", "1",
                           "--instance-initiated-shutdown-behavior", "terminate",
                           "--block-device-mappings", json.dumps([{"DeviceName": "/dev/sda1",
                                                                   "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True}}]),
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
