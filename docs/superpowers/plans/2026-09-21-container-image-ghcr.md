# Container Image & GHCR Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `von` into a CPU and a CUDA container image and publish both to GitHub Container Registry automatically on every push to `master`.

**Architecture:** One parameterized multi-stage `Dockerfile` produces both the CPU and the CUDA image. A `TORCH_BACKEND` build argument selects between the PyTorch CPU wheel and the CUDA wheel pinned in `uv.lock`; everything else in the recipe is shared. Only the **CPU** variant is published — it is built by a single-job GitHub Actions workflow and pushed to GHCR under two tags. GPU users build the CUDA variant locally from the same file. Model weights (~3.2 GB) are never baked in; they are fetched from the Hugging Face Hub into a mounted `HF_HOME` volume on first use.

**Tech Stack:** Docker (multi-stage, `python:3.12-slim-bookworm`), `uv` **0.12.17** (pinned via the official image), `uv.lock`, Docker Buildx (cache mounts), GitHub Actions (`docker/build-push-action@v6`), GHCR.

**Spec:** `docs/superpowers/specs/2026-09-21-container-image-ghcr-design.md`

## Global Constraints

- Base image is `python:3.12-slim-bookworm` for **both** variants. Do not introduce an `nvidia/cuda` base.
- `pyproject.toml`, `uv.lock`, and every file under `src/` are used as-is. Do not refactor them. You **may** edit them if this work uncovers a defect that requires it — but say so explicitly in the commit message rather than changing them silently.
- `torch` is pinned at `2.14.0` by `uv.lock`. The CPU image must end up with exactly `2.14.0+cpu`.
- The CPU image must contain **zero** distributions whose name starts with `nvidia`, `cuda`, or `triton`.
- `uv` is pinned to **0.12.17** by copying the binary from `ghcr.io/astral-sh/uv:0.12.17`. Do not replace this with the `curl … | sh` installer, and do not use a `latest` tag — that is the whole point of the pin.
- The build requires **BuildKit**: the `Dockerfile` uses `COPY --from=ghcr.io/astral-sh/uv:…` and `--mount=type=cache`. Docker 29 enables it by default; do not set `DOCKER_BUILDKIT=0`.
- Architecture is `linux/amd64` only. Do not add `linux/arm64`.
- The runtime user is `von`, uid **1001**, gid **1001**, and never root.
- `HF_HOME` is `/data/huggingface`; the container listens on **8000**.
- The image name is `ghcr.io/muhannadalrusayni/von` and **must be lowercased** in the workflow. GHCR rejects uppercase and the GitHub repository is `Muhannadalrusayni/von`.
- Published tags are exactly two: `<YYYY.MM.DD.HH.MM>` and `latest`. There is no `edge` tag, and no `-cuda` or `cuda-latest` tag — the CUDA variant is **not** published.
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

# Copy the uv binary from the official distroless image rather than running the
# curl installer: this pins the toolchain version and removes the apt layer.
# The binary is statically linked, so it runs on this glibc base.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependency layer first, so later source edits do not invalidate the torch install.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --no-dev --no-emit-project --no-hashes \
      --format requirements-txt -o /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python "${PYTHON_VERSION}" \
 && if [ "${TORCH_BACKEND}" = "cpu" ]; then \
      # uv.lock pins the CUDA-13 PyTorch wheel, whose ~19 nvidia-*/cuda-*/triton
      # dependencies are emitted as top-level pins in the export. The CPU wheel
      # needs none of them, so drop those lines before installing.
      # --no-hashes is required because the CPU wheel's digest differs from the
      # CUDA wheel recorded in the lock; versions stay exactly pinned.
      # --torch-backend is a `uv pip`-only feature; `uv sync` cannot express it.
      grep -vE '^(nvidia-|cuda[-_]|triton)' /tmp/requirements.txt > /tmp/requirements.cpu.txt; \
      uv pip install --python /opt/venv --torch-backend=cpu -r /tmp/requirements.cpu.txt; \
    else \
      uv pip install --python /opt/venv -r /tmp/requirements.txt; \
    fi

COPY src ./src
# --no-deps keeps the already-installed torch (possibly the CPU wheel) in place;
# without it, installing von-sdk would re-resolve torch from PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv --no-deps .

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

Three details in this file are load-bearing and must not be "simplified":
- `--no-deps` on the project install. Without it, `uv pip install .` re-resolves `torch` from PyPI and silently replaces the CPU wheel with the CUDA one.
- The `grep -vE` filter. `uv export` emits the `nvidia-*`, `cuda-*`, and `triton` pins as **top-level** requirements, so `--torch-backend=cpu` alone does not remove them.
- The `COPY --from=ghcr.io/astral-sh/uv:0.12.17` line, which is what pins the toolchain. `--torch-backend` exists only in the `uv pip` interface, and the `curl … | sh` installer resolves to whatever is current at build time — it produced uv 0.7.17 during design review while the current release was 0.12.17. Replacing the `COPY` re-introduces an unpinned input.

This exact file was built and verified during design review (spec §13): `torch==2.14.0+cpu`, zero CUDA distributions, console script on `PATH`, uid/gid 1001, `/health` OK, image size **993 MB**. If your result differs, stop and compare against the spec before changing anything.

- [ ] **Step 3: Verify the pinned uv image resolves, then build**

Run:
```bash
docker run --rm ghcr.io/astral-sh/uv:0.12.17 --version
```
Expected: `uv 0.12.17 (x86_64-unknown-linux-musl)`. If this tag does not resolve, the `COPY --from` in Step 2 fails too, so check here first.

Then run:
```bash
docker build --build-arg TORCH_BACKEND=cpu -t von:cpu .
```
Expected: build completes, exit code 0. A cold build takes roughly 3–4 minutes on this machine and downloads the CPU PyTorch wheel and its dependencies; with the cache mount warm, later builds are far quicker.

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
for i in $(seq 1 60); do
  curl -fsS --max-time 2 http://localhost:18000/health && break
  sleep 0.5
done
sleep 35
docker inspect von-smoke --format 'health={{.State.Health.Status}}'
docker rm -f von-smoke
```
Expected: the `{"status":"ok",...}` JSON, then `health=healthy`.

**Poll, do not `sleep`.** The server takes about **7 seconds** to become ready. A fixed 5- or 6-second sleep is a race that fails intermittently with `curl: (56) Recv failure: Connection reset by peer`; that exact failure was observed during design review and is what this loop replaces.

The health endpoint does not load the model, so this succeeds without the 3.2 GB weight download. Port 18000 is used on the host to avoid colliding with anything already on 8000.

- [ ] **Step 9: Verify one real decision request (bandwidth permitting)**

This step downloads roughly 3.2 GB of weights and is the only end-to-end check that the model actually loads and runs.

Run:
```bash
docker volume create von-hf
docker run --rm -v von-hf:/data/huggingface --entrypoint python von:cpu \
  -c "from huggingface_hub import snapshot_download; snapshot_download('wfzyx/von-1.0')"
docker run -d --name von-infer -p 18000:8000 -v von-hf:/data/huggingface von:cpu
for i in $(seq 1 60); do
  curl -fsS --max-time 2 http://localhost:18000/health >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS --max-time 600 -X POST http://localhost:18000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model":"von-1.0.0","state":{"error":"Disk volume /var/log at 98% capacity."},"questions":{"requires_intervention":{"type":"noul","instructions":"Does this disk space condition require operational intervention?"}}}'
docker rm -f von-infer
```
Expected: JSON containing an `answers` object whose `requires_intervention.noul` is a float between `0` and `1`.

The 600-second request timeout is deliberate: this first request loads ~1.5 GB of weights before answering, so a default timeout will abort it mid-load.

`curl -f` makes an HTTP 422 (the server's error path) fail the command, which is the behaviour we want.

> **Status: not performed.** Skipped during implementation — the ~3.2 GB download did not complete within the time budget. It is recorded as an accepted risk in spec §12.8 rather than silently marked as passed. Every other step in this task (build, CPU wheel, no CUDA packages, non-root, console script, `/health`, Docker healthcheck, image size) did pass.

- [ ] **Step 10: Record the image size**

Run:
```bash
docker image inspect von:cpu --format '{{.Size}}' | numfmt --to=iec
```
Expected: **993 MB**, the value measured during design review. A result in the multi-GB range means the CUDA wheel was installed into the "CPU" image — go back and re-check Step 5. Record the value for the pull request description.

- [ ] **Step 11: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "ci: add multi-stage Dockerfile with cpu and cuda variants"
```

---

### Task 2: Publish the CPU image to GHCR

Adds the workflow that builds the CPU variant and pushes two tags. The publish itself cannot run until this branch is merged to `master` (GitHub only dispatches `workflow_dispatch` for workflows present on the default branch), so verification here is static plus a local simulation of the tag-generation logic.

**Files:**
- Create: `.github/workflows/docker-publish.yml`

**Interfaces:**
- Consumes: the `TORCH_BACKEND` build arg from Task 1. CI passes `cpu` explicitly.
- Produces: GHCR tags `ghcr.io/muhannadalrusayni/von:<YYYY.MM.DD.HH.MM>` and `ghcr.io/muhannadalrusayni/von:latest`. Task 3 documents these exact strings and no others.

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
  build:
    name: Build and publish CPU image
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Compute image tags
        id: tags
        run: |
          set -euo pipefail
          # GHCR rejects uppercase names; the repository is MuhannadAlrusayni/von.
          image="${REGISTRY}/${GITHUB_REPOSITORY,,}"
          calver="$(date -u +%Y.%m.%d.%H.%M)"
          {
            echo "list<<TAGS_EOF"
            echo "${image}:${calver}"
            echo "${image}:latest"
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
            TORCH_BACKEND=cpu
          tags: ${{ steps.tags.outputs.list }}
          labels: |
            org.opencontainers.image.source=https://github.com/${{ github.repository }}
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.licenses=Apache-2.0
            org.opencontainers.image.title=von
            org.opencontainers.image.description=Von System One decision model (cpu)
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Three details here are load-bearing:
- `${GITHUB_REPOSITORY,,}` lowercases the repository name. Without it every push fails with "repository name must be lowercase".
- `TORCH_BACKEND=cpu` is passed explicitly even though it is also the `Dockerfile` default, so that changing that default cannot silently change what CI publishes.
- `org.opencontainers.image.source` must link the package to the repository. It only takes effect when present **before** the first publish.

- [ ] **Step 3: Verify the YAML parses and has the expected shape**

Run:
```bash
python3 -c "
import yaml
d = yaml.safe_load(open('.github/workflows/docker-publish.yml'))
assert set(d['jobs']) == {'build'}, d['jobs'].keys()
assert d['permissions'] == {'contents': 'read', 'packages': 'write'}, d['permissions']
triggers = d.get(True) or d.get('on')
assert set(triggers) == {'push', 'workflow_dispatch'}, triggers
args = d['jobs']['build']['steps'][-1]['with']['build-args']
assert args.strip() == 'TORCH_BACKEND=cpu', args
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

- [ ] **Step 5: Verify the tag-generation logic produces exactly two tags**

Run the step's shell block verbatim:

```bash
REGISTRY=ghcr.io GITHUB_REPOSITORY=Muhannadalrusayni/von bash -c '
set -euo pipefail
image="${REGISTRY}/${GITHUB_REPOSITORY,,}"
calver="$(date -u +%Y.%m.%d.%H.%M)"
echo "${image}:${calver}"
echo "${image}:latest"
'
```
Expected, exactly (the timestamp is the current UTC minute):
```
ghcr.io/muhannadalrusayni/von:<current UTC minute>
ghcr.io/muhannadalrusayni/von:latest
```

Confirm the output contains no `-cuda` tag.

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
git commit -m "ci: publish cpu image to ghcr on master"
```

- [ ] **Step 8: Post-merge verification (do not skip)**

This step cannot run until the branch is merged to `master`. After merging, run:
```bash
gh run list --workflow=docker-publish.yml --limit 1
gh run watch
```
Expected: a successful run. Then confirm the package and both tags exist:
```bash
gh api "/users/Muhannadalrusayni/packages/container/von/versions" --jq '.[].metadata.container.tags[]'
```
Expected: exactly the two tags from Step 5, and no `-cuda` tag.

This is the only place the publish path runs. The CUDA branch of the `Dockerfile` is not exercised anywhere — it is neither built here nor pushed by CI. See §12 of the spec.

---

### Task 3: Document container usage in the README

Adds a `### Container Image` subsection under `## Server Deployment (von serve)` documenting the published CPU image, the volume contract, and how to build the CUDA variant locally; plus a top-level `## Configuration` section documenting the environment variables the server, image, and clients read, with their defaults.

**Files:**
- Modify: `README.md` — insert immediately after the closing fence of the `### Wire Protocol Verification` block (currently line 349) and before the `---` on line 351.

**Interfaces:**
- Consumes: the image's runtime contract from Task 1 (`HF_HOME=/data/huggingface`, port 8000, entrypoint `von serve`, and the `TORCH_BACKEND` build arg) and the tag name from Task 2 (`latest`).
- Produces: nothing consumed by other tasks. This is the final task.

- [ ] **Step 1: Confirm the section does not already exist**

Run:
```bash
grep -n "Container Image" README.md
```
Expected: no output, exit code 1. This confirms Task 3 adds something genuinely absent.

- [ ] **Step 2: Insert the section**

In `README.md`, replace this text:

````
  }'
```

---

## Training Data & Domain Coverage
````

with this text:

````
  }'
```

### Container Image

A prebuilt CPU image is published for `linux/amd64`:

```bash
docker run --rm -p 8000:8000 -v von-hf:/data/huggingface \
  ghcr.io/muhannadalrusayni/von:latest
```

Weights (~3.2 GB) are not baked in; mount a volume at `HF_HOME`
(`/data/huggingface`) to persist them. To fetch them ahead of time:

```bash
docker run --rm -v von-hf:/data/huggingface --entrypoint python \
  ghcr.io/muhannadalrusayni/von:latest \
  -c "from huggingface_hub import snapshot_download; snapshot_download('wfzyx/von-1.0')"
```

Both variants come from the same `Dockerfile`; `TORCH_BACKEND=default` selects
CUDA instead of CPU. `--gpus all` is required, otherwise it falls back to CPU:

```bash
docker build --build-arg TORCH_BACKEND=default -t von:cuda .
docker run --rm --gpus all -p 8000:8000 -v von-hf:/data/huggingface von:cuda
```

See [Configuration](#configuration) for the environment variables the server and
image read.

---

## Configuration

The server, image, and clients read the following environment variables.
Command-line flags take precedence where both exist.

### Server and image

| Variable | Default | Description |
|---|---|---|
| `VON_BACKEND` | `option-marker` | Decision backend to load. `von serve --backend` overrides it; accepted values are `option-marker`, `modernbert`, `von-1.0`, `marker`, `laya`, `needle`, `berta-v3`. When the engine is used directly rather than through `von serve`, an unset value falls back to `von-1.0`. |
| `VON_DEVICE` | `auto` | Compute device: `auto`, `cuda`, `rocm`, `mps`, `dml`, `cpu`. `auto` prefers CUDA, then Apple MPS, then CPU. `von serve --device <x>` overrides it, except that passing `--device auto` leaves an existing value in place. |
| `VON_API_KEY` | unset | When set, `POST /v1/systemone` requires `Authorization: Bearer <VON_API_KEY>`. When unset, the endpoint is unauthenticated. |
| `HF_HOME` | `/data/huggingface` | Where model weights are cached; roughly 3.2 GB is fetched on first use. Mount a volume here to persist it across container replacements. Outside the image this follows the usual Hugging Face default, `~/.cache/huggingface`. |

### Clients

| Variable | Default | Description |
|---|---|---|
| `VON_BASE_URL` | `http://localhost:8000` | Server address used by the Python and TypeScript clients when not running in-process. |
| `TYPESAFE_BASE_URL` | unset | TypeScript client only: fallback for `VON_BASE_URL`. |
| `TYPESAFE_API_KEY` | unset | Fallback bearer token for both clients when `VON_API_KEY` is unset. |

---

## Training Data & Domain Coverage
````

- [ ] **Step 3: Verify the insertion landed correctly**

Run:
```bash
grep -n "Container Image" README.md
awk '/^## Server Deployment/,/^## Training Data/' README.md | grep -E '^#{2,4} '
```
Expected: `Container Image` appears once, and the headings in that range are, in order:

```
## Server Deployment (`von serve`)
### Wire Protocol Verification
### Container Image
## Configuration
### Server and image
### Clients
## Training Data & Domain Coverage
```

- [ ] **Step 3b: Verify configuration is not buried in the container section**

Run:
```bash
awk '/^### Container Image/,/^## Configuration/' README.md | grep -E 'VON_API_KEY|VON_BACKEND|VON_DEVICE'
```
Expected: no output. The container section must describe *running* the image, not
restate configuration; the variables belong to the `## Configuration` section.

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
Expected: exactly `ghcr.io/muhannadalrusayni/von:latest`. There must be **no** `cuda-latest` or `-cuda` reference anywhere, because nothing publishes such a tag — the CUDA instructions build a local `von:cuda` image instead.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document container image usage and ghcr tags"
```

---

## Completion Checklist

After all three tasks:

- [ ] `git log --oneline` shows the three implementation commits (`ci:`, `ci:`, `docs:`) on top of the design-spec and plan commits, which stay as they are
- [ ] `git diff --stat master` touches only `Dockerfile`, `.dockerignore`, `.github/workflows/docker-publish.yml`, `README.md`, and `docs/`
- [ ] `pyproject.toml`, `uv.lock`, and `src/` are untouched — unless a defect forced a change, in which case the commit message says so
- [ ] `docker run --rm --entrypoint python von:cpu -c "import torch; print(torch.__version__)"` prints `2.14.0+cpu`
- [ ] `docker image inspect von:cpu --format '{{.Size}}' | numfmt --to=iec` is ~993 MB, not multi-GB
- [ ] `Dockerfile` pins uv with `COPY --from=ghcr.io/astral-sh/uv:0.12.17`, and contains no `curl` installer and no `apt-get`
- [ ] The workflow passes `TORCH_BACKEND=cpu`, which is one of the two branches in the `Dockerfile`
- [ ] The workflow publishes exactly two tags (`<calver>`, `latest`) and no `-cuda` tag
- [ ] `README.md` documents both the published CPU pull and the local `--build-arg TORCH_BACKEND=default` CUDA build
- [ ] `grep -rn 'cuda-latest' README.md .github/ Dockerfile` returns nothing
- [ ] No occurrence of `TODO`, `TBD`, or `FIXME` in any new or modified file

## Deferred to after merge

The GHCR publish cannot be exercised on the feature branch. Task 2 Step 8 covers it, and it is the only step in this plan that requires the branch to have landed on `master`.
