# Container Image & GHCR Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `von` into a CPU and a CUDA container image and publish both to GitHub Container Registry automatically on every push to `master`.

**Architecture:** One parameterized multi-stage `Dockerfile` produces both variants. A `TORCH_BACKEND` build argument selects between the PyTorch CPU wheel and the CUDA wheel pinned in `uv.lock`; everything else in the recipe is shared. Model weights (~3.2 GB) are never baked in — they are fetched from the Hugging Face Hub into a mounted `HF_HOME` volume on first use. A GitHub Actions workflow builds both variants as a matrix and pushes four tags to GHCR.

**Tech Stack:** Docker (multi-stage, `python:3.12-slim-bookworm`), `uv` 0.7+, `uv.lock`, Docker Buildx, GitHub Actions (`docker/build-push-action@v6`), GHCR.

**Spec:** `docs/superpowers/specs/2026-09-21-container-image-ghcr-design.md`

## Global Constraints

- Base image is `python:3.12-slim-bookworm` for **both** variants. Do not introduce an `nvidia/cuda` base.
- `pyproject.toml`, `uv.lock`, and every file under `src/` must **not** be modified. The image is built from them as-is.
- `torch` is pinned at `2.14.0` by `uv.lock`. The CPU image must end up with exactly `2.14.0+cpu`.
- The CPU image must contain **zero** distributions whose name starts with `nvidia`, `cuda`, or `triton`.
- Architecture is `linux/amd64` only. Do not add `linux/arm64`.
- The runtime user is `von`, uid **1001**, gid **1001**, and never root.
- `HF_HOME` is `/data/huggingface`; the container listens on **8000**.
- The image name is `ghcr.io/muhannadalrusayni/von` and **must be lowercased** in the workflow. GHCR rejects uppercase and the GitHub repository is `Muhannadalrusayni/von`.
- Published tags are exactly: `<YYYY.MM.DD.HH.MM>`, `<YYYY.MM.DD.HH.MM>-cuda`, `latest`, `cuda-latest`. There is no `edge` and no `:cuda` tag.
- Timestamps are UTC, computed **once** per workflow run.
- The workflow publishes only on `push` to `master` and on `workflow_dispatch`.
- The words `TODO`, `TBD`, and `FIXME` must not appear in any artifact.
- Commits follow the repository's existing conventional-commit convention (`ci:`, `docs:`, `feat:`).

---

### Task 1: Build the CPU container image

Produces a runnable image with a working `von serve` entrypoint, the CPU PyTorch wheel, and no CUDA packages. This is the only task whose deliverable can be fully verified on this machine.

**Files:**
- Create: `.dockerignore`
- Create: `Dockerfile`

**Interfaces:**
- Consumes: nothing. This is the first task.
- Produces: a Docker build context that accepts build arg `TORCH_BACKEND` with values `cpu` or `default`; an image whose entrypoint is `von serve`, whose default command is `--host 0.0.0.0 --port 8000 --backend option-marker`, which exposes port `8000`, sets `HF_HOME=/data/huggingface`, runs as uid/gid `1001`, and contains a `von-sdk` console script on `PATH`. Task 2 depends on the exact build-arg name and its two accepted values.

- [ ] **Step 1: Create `.dockerignore`**

Create `.dockerignore` at the repository root with exactly this content:

```
.git
.github
.venv
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
build/
dist/
checkpoints/
data/
assets/
benchmarks/
examples/
tests/
training/
js/
docs/
```

Do **not** exclude `README.md` or other markdown: they are ~30 KB total, and excluding them risks a `hatchling` build-metadata surprise for no meaningful gain.

- [ ] **Step 2: Create `Dockerfile`**

Create `Dockerfile` at the repository root with exactly this content:

```dockerfile
# syntax=docker/dockerfile:1

# Von container image.
#
# Two variants differ only in which PyTorch wheel is installed:
#   docker build --build-arg TORCH_BACKEND=cpu     -t von:cpu  .
#   docker build --build-arg TORCH_BACKEND=default -t von:cuda .
#
# Model weights (~3.2 GB) are NOT baked in; they are fetched from the Hugging
# Face Hub into HF_HOME on first use. Mount a volume there to persist them.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG PYTHON_VERSION
ARG TORCH_BACKEND=cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /build

# Dependency layer first, so later source edits do not invalidate the torch install.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes \
      --format requirements-txt -o /tmp/requirements.txt

RUN uv venv /opt/venv --python "${PYTHON_VERSION}" \
 && if [ "${TORCH_BACKEND}" = "cpu" ]; then \
      # uv.lock pins the CUDA-13 PyTorch wheel, whose ~19 nvidia-*/cuda-*/triton
      # dependencies are emitted as top-level pins in the export. The CPU wheel
      # needs none of them, so drop those lines before installing.
      # --no-hashes is required because the CPU wheel's digest differs from the
      # CUDA wheel recorded in the lock; versions stay exactly pinned.
      grep -vE '^(nvidia-|cuda[-_]|triton)' /tmp/requirements.txt > /tmp/requirements.cpu.txt; \
      uv pip install --python /opt/venv --torch-backend=cpu -r /tmp/requirements.cpu.txt; \
    else \
      uv pip install --python /opt/venv -r /tmp/requirements.txt; \
    fi

COPY src ./src
# --no-deps keeps the already-installed torch (possibly the CPU wheel) in place;
# without it, installing von-sdk would re-resolve torch from PyPI.
RUN uv pip install --python /opt/venv --no-deps .

# --------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG PYTHON_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/data/huggingface \
    VON_BACKEND=option-marker

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 1001 von \
 && useradd --uid 1001 --gid 1001 --shell /usr/sbin/nologin --create-home von \
 && mkdir -p "${HF_HOME}" \
 && chown -R von:von /data

USER von
WORKDIR /home/von

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["von", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--backend", "option-marker"]
```

Two details in this file are load-bearing and must not be "simplified":
- `--no-deps` on the project install. Without it, `uv pip install .` re-resolves `torch` from PyPI and silently replaces the CPU wheel with the CUDA one.
- The `grep -vE` filter. `uv export` emits the `nvidia-*`, `cuda-*`, and `triton` pins as **top-level** requirements, so `--torch-backend=cpu` alone does not remove them.

- [ ] **Step 3: Verify the image builds**

Run:
```bash
docker build --build-arg TORCH_BACKEND=cpu -t von:cpu .
```
Expected: build completes, exit code 0. The first run downloads roughly 500 MB of wheels and can take several minutes.

- [ ] **Step 4: Verify PyTorch is the CPU wheel**

Run:
```bash
docker run --rm --entrypoint python von:cpu -c "import torch; print(torch.__version__)"
```
Expected: `2.14.0+cpu`. If it prints `2.14.0` without the `+cpu` suffix, `--no-deps` is missing or `--torch-backend=cpu` did not take effect — stop and fix Step 2.

- [ ] **Step 5: Verify no CUDA packages are present**

Run:
```bash
docker run --rm --entrypoint python von:cpu -c "import importlib.metadata as m; print(sorted(d.metadata['Name'] for d in m.distributions() if d.metadata['Name'].lower().startswith(('nvidia','cuda','triton'))))"
```
Expected: `[]`. A non-empty list means the `grep -vE` filter is wrong.

- [ ] **Step 6: Verify the CLI and package are installed on `PATH`**

Run:
```bash
docker run --rm von:cpu --help
```
Expected: output containing `Usage: von serve`.

Then run:
```bash
docker run --rm --entrypoint python von:cpu -c "import von; print(von.__file__)"
```
Expected: a path under `/opt/venv/`, proving the project installed into the venv rather than falling back to a source import.

- [ ] **Step 7: Verify the container does not run as root**

Run:
```bash
docker run --rm --entrypoint id von:cpu
```
Expected: `uid=1001(von) gid=1001(von)`.

- [ ] **Step 8: Verify the server starts and answers `/health`**

Run:
```bash
docker run -d --name von-smoke -p 18000:8000 von:cpu
sleep 5
curl -fsS http://localhost:18000/health
docker rm -f von-smoke
```
Expected: JSON containing `"status":"ok"` and `"service":"von-decision-server"`.

The health endpoint does not load the model, so this succeeds without the 3.2 GB weight download. Port 18000 is used on the host to avoid colliding with anything already on 8000.

- [ ] **Step 9: Verify one real decision request (bandwidth permitting)**

This step downloads roughly 3.2 GB of weights and is the only end-to-end check that the model actually loads and runs.

Run:
```bash
docker volume create von-hf
docker run --rm -v von-hf:/data/huggingface --entrypoint python von:cpu \
  -c "from huggingface_hub import snapshot_download; snapshot_download('wfzyx/von-1.0')"
docker run -d --name von-infer -p 18000:8000 -v von-hf:/data/huggingface von:cpu
sleep 10
curl -fsS -X POST http://localhost:18000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model":"von-1.0.0","state":{"error":"Disk volume /var/log at 98% capacity."},"questions":{"requires_intervention":{"type":"noul","instructions":"Does this disk space condition require operational intervention?"}}}'
docker rm -f von-infer
```
Expected: JSON containing an `answers` object whose `requires_intervention.noul` is a float between `0` and `1`.

`curl -f` makes an HTTP 422 (the server's error path) fail the command, which is the behaviour we want. If bandwidth or disk space preclude this step, record that explicitly rather than marking it passed.

- [ ] **Step 10: Record the image size**

Run:
```bash
docker image inspect von:cpu --format '{{.Size}}' | numfmt --to=iec
```
Expected: a value in the low single-digit GB. Record it for the pull request description.

- [ ] **Step 11: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "ci: add multi-stage Dockerfile with cpu and cuda variants"
```

---

### Task 2: Publish both variants to GHCR

Adds the workflow that builds the CPU and CUDA variants and pushes four tags. The publish itself cannot run until this branch is merged to `master` (GitHub only dispatches `workflow_dispatch` for workflows present on the default branch), so verification here is static plus a local simulation of the tag-generation logic.

**Files:**
- Create: `.github/workflows/docker-publish.yml`

**Interfaces:**
- Consumes: the `TORCH_BACKEND` build arg from Task 1, with values `cpu` and `default`.
- Produces: GHCR tags `ghcr.io/muhannadalrusayni/von:<YYYY.MM.DD.HH.MM>`, `…<YYYY.MM.DD.HH.MM>-cuda`, `…:latest`, and `…:cuda-latest`. Task 3 documents these exact strings.

- [ ] **Step 1: Confirm the check currently fails**

Run:
```bash
ls .github/workflows/
```
Expected: `publish-hf-weights.yml`, `sync-hf-card.yml`, `test.yml`. There is no `docker-publish.yml`, which confirms Task 2 adds something genuinely absent.

- [ ] **Step 2: Create `.github/workflows/docker-publish.yml`**

Create `.github/workflows/docker-publish.yml` with exactly this content:

```yaml
name: Build and Publish Image

on:
  push:
    branches: [master]
    paths-ignore:
      - '**.md'
      - 'docs/**'
      - 'assets/**'
  workflow_dispatch:

permissions:
  contents: read
  packages: write

concurrency:
  group: image-${{ github.ref }}
  cancel-in-progress: false

env:
  REGISTRY: ghcr.io

jobs:
  meta:
    name: Resolve release metadata
    runs-on: ubuntu-latest
    outputs:
      calver: ${{ steps.calver.outputs.tag }}
    steps:
      - name: Compute CalVer tag
        id: calver
        run: echo "tag=$(date -u +%Y.%m.%d.%H.%M)" >> "$GITHUB_OUTPUT"

  build:
    name: Build ${{ matrix.variant }}
    needs: meta
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - variant: cpu
            torch_backend: cpu
            suffix: ""
            extra_tags: "latest"
          - variant: cuda
            torch_backend: default
            suffix: "-cuda"
            extra_tags: "cuda-latest"
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Compute image tags
        id: tags
        env:
          CALVER: ${{ needs.meta.outputs.calver }}
          SUFFIX: ${{ matrix.suffix }}
          EXTRA_TAGS: ${{ matrix.extra_tags }}
        run: |
          set -euo pipefail
          # GHCR rejects uppercase names; the repository is MuhannadAlrusayni/von.
          image="${REGISTRY}/${GITHUB_REPOSITORY,,}"
          {
            echo "list<<TAGS_EOF"
            echo "${image}:${CALVER}${SUFFIX}"
            for tag in ${EXTRA_TAGS}; do
              echo "${image}:${tag}"
            done
            echo "TAGS_EOF"
          } >> "$GITHUB_OUTPUT"

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          build-args: |
            TORCH_BACKEND=${{ matrix.torch_backend }}
          tags: ${{ steps.tags.outputs.list }}
          labels: |
            org.opencontainers.image.source=https://github.com/${{ github.repository }}
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.licenses=Apache-2.0
            org.opencontainers.image.title=von
            org.opencontainers.image.description=Von System One decision model (${{ matrix.variant }})
          cache-from: type=gha,scope=${{ matrix.variant }}
          cache-to: type=gha,mode=max,scope=${{ matrix.variant }}
```

Three details here are load-bearing:
- `${GITHUB_REPOSITORY,,}` lowercases the repository name. Without it every push fails with "repository name must be lowercase".
- CalVer is computed once in the `meta` job. Computing it inside the matrix would let the CPU and CUDA images of one release land on different minutes.
- `org.opencontainers.image.source` must link the package to the repository. It only takes effect when present **before** the first publish.

- [ ] **Step 3: Verify the YAML parses and has the expected shape**

Run:
```bash
python3 -c "
import yaml
d = yaml.safe_load(open('.github/workflows/docker-publish.yml'))
assert set(d['jobs']) == {'meta', 'build'}, d['jobs'].keys()
assert d['permissions'] == {'contents': 'read', 'packages': 'write'}, d['permissions']
m = d['jobs']['build']['strategy']['matrix']['include']
assert [x['variant'] for x in m] == ['cpu', 'cuda'], m
assert [x['torch_backend'] for x in m] == ['cpu', 'default'], m
assert [x['extra_tags'] for x in m] == ['latest', 'cuda-latest'], m
print('workflow shape OK')
"
```
Expected: `workflow shape OK`.

- [ ] **Step 4: Verify with actionlint if available**

Run:
```bash
command -v actionlint >/dev/null && actionlint .github/workflows/docker-publish.yml || echo "actionlint not installed; skipping"
```
Expected: either no output from `actionlint` (clean), or the explicit skip message. If `actionlint` is installed and reports errors, fix them before continuing.

- [ ] **Step 5: Verify the tag-generation logic produces exactly four tags**

Run the step's shell block verbatim against representative matrix values:

```bash
REGISTRY=ghcr.io GITHUB_REPOSITORY=Muhannadalrusayni/von CALVER=2026.09.21.20.47 bash -c '
for v in cpu cuda; do
  case "$v" in
    cpu)  SUFFIX="";      EXTRA_TAGS="latest" ;;
    cuda) SUFFIX="-cuda"; EXTRA_TAGS="cuda-latest" ;;
  esac
  image="${REGISTRY}/${GITHUB_REPOSITORY,,}"
  echo "${image}:${CALVER}${SUFFIX}"
  for tag in ${EXTRA_TAGS}; do echo "${image}:${tag}"; done
done'
```
Expected, exactly:
```
ghcr.io/muhannadalrusayni/von:2026.09.21.20.47
ghcr.io/muhannadalrusayni/von:latest
ghcr.io/muhannadalrusayni/von:2026.09.21.20.47-cuda
ghcr.io/muhannadalrusayni/von:cuda-latest
```

- [ ] **Step 6: Verify the CalVer format and lowercasing**

Run:
```bash
echo "$(date -u +%Y.%m.%d.%H.%M)"
```
Expected: matches `^[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}$`.

Then run:
```bash
GITHUB_REPOSITORY=Muhannadalrusayni/von bash -c 'echo "${GITHUB_REPOSITORY,,}"'
```
Expected: `muhannadalrusayni/von`.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/docker-publish.yml
git commit -m "ci: build and publish cpu and cuda images to ghcr on master"
```

- [ ] **Step 8: Post-merge verification (do not skip)**

This step cannot run until the branch is merged to `master`. After merging, run:
```bash
gh run list --workflow=docker-publish.yml --limit 1
gh run watch
```
Expected: a successful run. Then confirm both packages and all four tags exist:
```bash
gh api "/users/Muhannadalrusayni/packages/container/von/versions" --jq '.[].metadata.container.tags[]'
```
Expected: the four tags from Step 5.

If the run fails on the CUDA leg, the cause is the `default` branch of the install step in Task 1 — the CPU leg passing does not exercise it.

---

### Task 3: Document container usage in the README

Adds a `### Container Image` subsection to the existing `## Server Deployment (von serve)` section, documenting the volume contract and both variants.

**Files:**
- Modify: `README.md` — insert immediately after the closing fence of the `### Wire Protocol Verification` block (currently line 349) and before the `---` on line 351.

**Interfaces:**
- Consumes: the image's runtime contract from Task 1 (`HF_HOME=/data/huggingface`, port 8000, entrypoint `von serve`) and the tag names from Task 2 (`latest`, `cuda-latest`).
- Produces: nothing consumed by other tasks. This is the final task.

- [ ] **Step 1: Confirm the section does not already exist**

Run:
```bash
grep -n "Container Image" README.md
```
Expected: no output, exit code 1. This confirms Task 3 adds something genuinely absent.

- [ ] **Step 2: Insert the section**

In `README.md`, replace this text:

```
  }'
```

---

## Training Data & Domain Coverage
```

with this text:

```
  }'
```

### Container Image

Prebuilt images are published to GitHub Container Registry for `linux/amd64`:

| Variant | Tag | Notes |
|---|---|---|
| CPU | `ghcr.io/muhannadalrusayni/von:latest` | Runs anywhere; no GPU required |
| CUDA | `ghcr.io/muhannadalrusayni/von:cuda-latest` | Requires the host NVIDIA driver and `--gpus all` |

Model weights (~3.2 GB) are **not** baked into the image. They are downloaded
from the Hugging Face Hub on first use into `HF_HOME` (`/data/huggingface`).
Mount a volume there so the download survives container replacement:

```bash
# One-time: fetch the weights into the named volume (~3.2 GB).
docker run --rm -v von-hf:/data/huggingface --entrypoint python \
  ghcr.io/muhannadalrusayni/von:latest \
  -c "from huggingface_hub import snapshot_download; snapshot_download('wfzyx/von-1.0')"

# Serve on http://localhost:8000
docker run --rm -p 8000:8000 -v von-hf:/data/huggingface \
  ghcr.io/muhannadalrusayni/von:latest

# CUDA variant
docker run --rm --gpus all -p 8000:8000 -v von-hf:/data/huggingface \
  ghcr.io/muhannadalrusayni/von:cuda-latest
```

Skipping the pre-warm step is fine — the first `/v1/systemone` request triggers
the download, but that request blocks until it completes. Allow roughly 4 GB of
free space for the volume.

Set `VON_API_KEY` to require `Authorization: Bearer <key>` on `/v1/systemone`.
The default backend is `option-marker`; override it with `VON_BACKEND` or the
`--backend` flag. Override the compute device with `VON_DEVICE` (`auto`,
`cuda`, `rocm`, `mps`, `dml`, `cpu`).

---

## Training Data & Domain Coverage
```

- [ ] **Step 3: Verify the insertion landed correctly**

Run:
```bash
grep -n "Container Image" README.md
awk '/^## Server Deployment/,/^## Training Data/' README.md | grep -nE '^#{2,3} '
```
Expected: `Container Image` appears once; the heading order within the range is `## Server Deployment (von serve)`, `### Wire Protocol Verification`, `### Container Image`.

- [ ] **Step 4: Verify the fenced code blocks are balanced**

Run:
```bash
python3 -c "
lines = open('README.md').read().splitlines()
fences = [i for i, l in enumerate(lines) if l.strip().startswith('\`\`\`')]
assert len(fences) % 2 == 0, f'unbalanced fences: {len(fences)}'
print(f'{len(fences)} fence markers, balanced')
"
```
Expected: an even number of fence markers. An odd count means a code block was left open, which would break the rest of the document's rendering.

- [ ] **Step 5: Verify the documented tags match the workflow**

Run:
```bash
grep -oE 'ghcr\.io/muhannadalrusayni/von:[a-z0-9.-]+' README.md | sort -u
```
Expected: `ghcr.io/muhannadalrusayni/von:cuda-latest` and `ghcr.io/muhannadalrusayni/von:latest` — both of which must also appear in `.github/workflows/docker-publish.yml` as values of `extra_tags`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document container image usage and ghcr tags"
```

---

## Completion Checklist

After all three tasks:

- [ ] `git log --oneline` shows three new commits on `ci/container-image-ghcr` beyond the spec commits
- [ ] `git diff --stat master` touches only `Dockerfile`, `.dockerignore`, `.github/workflows/docker-publish.yml`, `README.md`, and `docs/`
- [ ] `pyproject.toml`, `uv.lock`, and `src/` are untouched
- [ ] `docker run --rm --entrypoint python von:cpu -c "import torch; print(torch.__version__)"` prints `2.14.0+cpu`
- [ ] The workflow's `TORCH_BACKEND` values (`cpu`, `default`) match the branches in the `Dockerfile`
- [ ] No occurrence of `TODO`, `TBD`, or `FIXME` in any new or modified file

## Deferred to after merge

The GHCR publish cannot be exercised on the feature branch. Task 2 Step 8 covers it, and it is the only step in this plan that requires the branch to have landed on `master`.
