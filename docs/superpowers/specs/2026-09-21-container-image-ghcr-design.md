# Container Image & GHCR Publishing — Design

- **Date:** 2026-09-21
- **Branch:** `ci/container-image-ghcr`
- **Repository:** `muhannadalrusayni/von` (fork of `wfzyx/von`)
- **Status:** Approved. Implementation plan written at
  `docs/superpowers/plans/2026-09-21-container-image-ghcr.md`; the CPU image has
  been built and verified (§13) but no plan task has been executed.

## 1. Problem

`von` currently has no container image. Running it requires cloning the
repository, installing `uv`, resolving `uv.lock`, and downloading ~3.2 GB of
model weights. There is no build or publish automation for a container, so
there is no way to consume the project as a runtime dependency.

This design adds a reproducible container build and publishes it to the GitHub
Container Registry (GHCR).

## 2. Scope

**In scope**

- A `Dockerfile` that builds the project into a runnable image.
- Two variants: CPU and CUDA.
- A GitHub Actions workflow that builds both variants and publishes them to
  `ghcr.io/muhannadalrusayni/von` with CalVer tags.

**Out of scope**

- Refactoring `pyproject.toml`, `uv.lock`, or application source for reasons
  unrelated to this work. Editing them *is* permitted when required to fix a
  defect this work uncovers.
- Baked-in model weights.
- Multi-architecture builds (arm64).
- Kubernetes manifests, Helm charts, or Compose files.
- Changing application behaviour or fixing existing bugs.

## 3. Context and constraints discovered

These were verified against the repository and the upstream registry, not
assumed.

### Application

- Python `3.12` (`.python-version`, `requires-python = ">=3.12"`), managed by
  `uv` (`uv.lock`, revision 3), packaged by `hatchling`.
- Distribution name `von-sdk`; console script `von = "von.cli:main"`.
- `von serve` starts a FastAPI/uvicorn server via `von.server:app`.
- HTTP surface: `GET /` and `GET /health`, `GET /v1/models`,
  `POST /v1/systemone`.
- Default backend for `serve` is `option-marker`; the engine-level default
  (`VON_BACKEND`) is `von-1.0`.
- Environment variables read by the application: `VON_BACKEND`, `VON_DEVICE`,
  `VON_API_KEY` (optional bearer auth), `VON_BASE_URL`, `TYPESAFE_API_KEY`
  (client-side only).

### Model weights

- Weights are **not** in the repository; `.gitignore` excludes `checkpoints/`.
- `OptionMarkerBackend` looks first for `checkpoints/von-option-marker/option_marker.pt`
  — a **relative** path with no environment override — and otherwise downloads
  from the Hugging Face Hub repository `wfzyx/von-1.0`.
- The upstream Hub repository totals **3.17 GB**: `model.safetensors` (1583 MB),
  `option_marker.pt` (1581 MB), plus tokenizer and config files.
- `OptionMarkerModel` calls `AutoModel.from_pretrained(base_model_id)`, so the
  runtime needs *both* `model.safetensors` and `option_marker.pt`.

### PyTorch

- `uv.lock` pins `torch==2.14.0` from `https://pypi.org/simple`. That is the
  **CUDA 13** wheel (554 MB on x86_64) and it declares 15 `nvidia-*`
  packages, `cuda-bindings`, `cuda-toolkit`, and `triton` — 19 CUDA-only
  distributions in total.
- A plain `uv sync` therefore produces a CUDA image even for a "CPU" variant.

### Toolchain

- `--torch-backend` is documented as a `uv pip`-only feature: "At present,
  `--torch-backend` is only available in the `uv pip` interface." `uv sync`
  therefore cannot express the CPU selection, which is what forces the
  `uv export` + `uv pip` strategy in §6 rather than a stylistic preference.
- The official Docker guidance recommends copying the uv binary from
  `ghcr.io/astral-sh/uv:<pinned>` and warns against an unpinned install. The
  `curl … | sh` installer resolved to **uv 0.7.17** on the development machine
  while the current release is **0.12.17** — five minor versions apart.
- Verified present in `ghcr.io/astral-sh/uv:0.12.17`: `uv export --locked` and
  `--frozen`, `uv pip install --torch-backend`, and `uv pip install --python`.
  The image is `x86_64-unknown-linux-musl`; the binary is statically linked and
  runs on this glibc base, which is the pattern the official guide documents.
- `python:3.12-slim-bookworm` already contains `ca-certificates`, so the
  official `COPY --from=ghcr.io/astral-sh/uv:...` pattern does not break TLS.

### CI and registry

- Existing workflows: `test.yml`, `publish-hf-weights.yml`, `sync-hf-card.yml`.
  No container workflow exists.
- The GitHub repository owner is `MuhannadAlrusayni`. GHCR requires lowercase
  image names, so `GITHUB_REPOSITORY` must be lowercased or every push fails.
- `workflow_dispatch` only triggers for workflows present on the **default**
  branch.

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Ship two variants: CPU and CUDA | User selection. CUDA gives the sub-18 ms latency claimed in the README; CPU runs anywhere. |
| D2 | Build `linux/amd64` only | CUDA is amd64-only regardless. arm64 under QEMU adds 20–40 min per build with no meaningful speed signal. |
| D3 | Thin images; weights downloaded at first run into a mounted volume | Avoids adding 3.2 GB of weights to *each* variant, and keeps the weight artifact out of the registry entirely. |
| D4 | One parameterized multi-stage `Dockerfile` | Both variants share ~90% of the recipe; only the PyTorch install differs. One file means the security settings cannot drift between variants. |
| D5 | CPU variant installs PyTorch via `uv export` + CUDA-package filter + `--torch-backend=cpu` | Verified against the real lock: yields `torch==2.14.0+cpu` with zero CUDA packages. Neither `pyproject.toml` nor `uv.lock` is modified. |
| D6 | Both variants use the same `python:3.12-slim-bookworm` base | The `nvidia-*` pip packages bundle their own CUDA runtime libraries, so the CUDA image needs only the **host** driver. This is what makes one file viable. |
| D7 | Runtime runs as non-root uid/gid 1001 | Standard container hardening; no application code requires root. |
| D8 | CalVer tags, computed once per run in UTC | Per the requested `YYYY.MM.DD.HH.MM` scheme. |
| D9 | Publish on push to `master`, plus `workflow_dispatch` | Automatic master builds with a manual escape hatch. |
| D10 | Add a short "Container image" section to `README.md` | An unpublished usage contract is not useful; the image is unusable to a reader without the volume/env/GPU invocation. Confirmed at spec review. |
| D11 | Pin the uv toolchain by copying the binary from `ghcr.io/astral-sh/uv:0.12.17` | The `curl … \| sh` installer resolved to uv 0.7.17 on the development machine while the current release is 0.12.17, so an installer-based build pins nothing. Verified that the tag exists and that the pinned version supports every flag used here. |

## 5. Files

| Path | Change | Purpose |
|---|---|---|
| `Dockerfile` | New | Parameterized multi-stage build |
| `.dockerignore` | New | Keeps the build context small |
| `.github/workflows/docker-publish.yml` | New | Build matrix and GHCR publishing |
| `docs/superpowers/specs/2026-09-21-container-image-ghcr-design.md` | New | This document |
| `README.md` | Modified | Short "Container image" usage section (D10) |

No existing source file, `pyproject.toml`, or `uv.lock` is modified. These may be
edited if this work uncovers a defect that requires it; no such defect is known
at the time of writing.

## 6. Dockerfile

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

### Notes on the build

- **uv is pinned by copying the binary from `ghcr.io/astral-sh/uv:0.12.17`.** See
  §3 for why the `curl … | sh` installer was rejected. This also removes the
  `apt-get` install of `curl` and `ca-certificates`; the base image already ships
  a CA bundle (verified: 224 KB at `/etc/ssl/certs/ca-certificates.crt`), so no
  TLS capability is lost.
- **`--locked`, not `--frozen`.** This is not a workspace. `--locked` fails the
  build when `uv.lock` is stale; `--frozen` would silently accept it.
- **The uv cache is a BuildKit cache mount.** `--mount=type=cache,target=/root/.cache/uv`
  keeps downloaded wheels between builds, which matters because any change to
  `uv.lock` otherwise re-downloads the CPU wheel. `UV_LINK_MODE=copy` is required
  alongside it, since the cache and the sync target are on different filesystems.
- **`--no-hashes` is a deliberate trade-off.** The CPU wheel and the CUDA wheel
  are different artifacts with different digests, so the lock's hashes cannot
  apply to both. Versions remain exactly pinned by the export. This is the only
  loss of integrity checking and it is documented inline.
- **`--no-deps` on the project install is load-bearing.** Without it, installing
  `von-sdk` re-resolves `torch` from PyPI and silently replaces the CPU wheel
  with the CUDA one.
- **No `VOLUME` instruction.** Declaring `VOLUME /data/huggingface` would create
  anonymous volumes that silently discard the 3.2 GB cache on container
  replacement. The volume is documented instead.

## 7. `.dockerignore`

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

`README.md` and other markdown are intentionally **not** excluded: they total
~30 KB and excluding them risks a `hatchling` build-metadata surprise for
negligible benefit.

## 8. Workflow

`.github/workflows/docker-publish.yml`:

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

### Notes on the workflow

- **CalVer is computed once**, in the `meta` job, and passed to both matrix legs.
  Computing it per-leg would let the CPU and CUDA images of the same release land
  on different minutes.
- **`${GITHUB_REPOSITORY,,}` is required**, not cosmetic — GHCR rejects uppercase
  repository names and every push would fail without it.
- `org.opencontainers.image.source` links the GHCR package to the repository,
  which is what grants repository-scoped permissions over the package.
- `type=gha` caching is scoped per variant so the CPU and CUDA torch layers do
  not poison each other's cache.

## 9. Tag matrix

| Variant | Build arg | Immutable tag | Moving tags |
|---|---|---|---|
| CPU | `TORCH_BACKEND=cpu` | `ghcr.io/muhannadalrusayni/von:<YYYY.MM.DD.HH.MM>` | `:latest` |
| CUDA | `TORCH_BACKEND=default` | `ghcr.io/muhannadalrusayni/von:<YYYY.MM.DD.HH.MM>-cuda` | `:cuda-latest` |

Timestamps are UTC, derived from build time via `date -u`. The CUDA variant
publishes **no** bare `:cuda` tag; its floating "latest" is named `:cuda-latest`.

The scheme is intentionally asymmetric: `:latest` and `:cuda-latest` are the two
tags a consumer is most likely to guess. Confirmed at spec review — see §14.

## 10. Runtime contract

| Aspect | Value |
|---|---|
| Port | `8000` (TCP) |
| Entrypoint | `von serve` |
| Default args | `--host 0.0.0.0 --port 8000 --backend option-marker` |
| Persistent path | `/data/huggingface` (`HF_HOME`) — mount a volume here |
| User | `von`, uid/gid `1001` |
| Healthcheck | `GET /health` every 30 s, 15 s start period, 3 retries |
| GPU (CUDA variant) | Requires the host driver and `docker run --gpus all` |
| Auth | Optional; set `VON_API_KEY` to require `Authorization: Bearer <key>` |
| Backend override | `VON_BACKEND` |
| Device override | `VON_DEVICE` (`auto`, `cuda`, `rocm`, `mps`, `dml`, `cpu`) |

### Usage

```bash
# One-time (or volume-persisted) weight download: ~3.2 GB
docker run --rm -v von-hf:/data/huggingface --entrypoint python \
  ghcr.io/muhannadalrusayni/von:latest \
  -c "from huggingface_hub import snapshot_download; snapshot_download('wfzyx/von-1.0')"

# Serve
docker run --rm -p 8000:8000 -v von-hf:/data/huggingface \
  ghcr.io/muhannadalrusayni/von:latest

# CUDA variant
docker run --rm --gpus all -p 8000:8000 -v von-hf:/data/huggingface \
  ghcr.io/muhannadalrusayni/von:cuda-latest
```

## 11. Error handling and failure modes

| Failure | Behaviour | Mitigation |
|---|---|---|
| No network on first request | `OptionMarkerBackend` raises `RuntimeError`; the server maps it to HTTP 422 with the message | Pre-warm the volume; the error text names the Hub repo |
| Weights not persisted | 3.2 GB re-downloaded per container | Mount `/data/huggingface` |
| Insufficient disk | Download fails mid-write | Document ~4 GB free space requirement |
| CUDA variant without a GPU | Server starts on CPU; `--device auto` falls back | Document; use the CPU variant |
| CUDA variant without `--gpus all` | Starts, runs on CPU — silent slow path | Document explicitly |
| CPU wheel replaced by CUDA wheel | Would silently inflate the image | Prevented by `--no-deps`; asserted in verification |
| Two pushes within one minute | Identical CalVer tag; the later overwrites the earlier | Accepted; see §12 |

## 12. Known limitations and accepted risks

1. **GHCR publishing cannot be end-to-end verified before merge.** GitHub only
   dispatches `workflow_dispatch` for workflows on the default branch, and this
   workflow is introduced on a feature branch. The first real publish happens
   after merge to `master`. The Dockerfile itself is fully verifiable locally.
2. **Minute-resolution CalVer collides.** Two merges in the same UTC minute
   produce the same tag and the second silently overwrites the first. Second-level
   precision was not requested.
3. **The CUDA image is large.** The `nvidia-*`, `cuda-*`, and `triton` pip
   packages add several GB over the CPU image. The exact size is deliberately not
   claimed here: the CUDA variant is not built locally (§13). This is inherent to
   supporting a CUDA wheel on a shared slim base.
4. **`:latest` points at CPU, `:cuda-latest` at CUDA.** A consumer assuming
   `:latest` is GPU-capable gets the CPU build.
5. **The healthcheck does not imply the model is loaded.** The engine loads
   weights lazily on the first `/v1/systemone` request, so `/health` returns 200
   before any inference is possible. The pre-warm command in §10 is the remedy.
6. **`--no-hashes`** for the CPU variant, as discussed in §6.
7. **The CUDA-dependency filter is maintained by hand.** The
   `grep -vE '^(nvidia-|cuda[-_]|triton)'` step is not derived from the lock; it
   must be extended if `uv.lock` ever gains a CUDA-only dependency that does not
   match those prefixes. This is the cost of not adopting uv's `tool.uv.sources`
   extra-based split, which would require editing `pyproject.toml` and
   regenerating `uv.lock`.

## 13. Verification plan

Executed against a real Docker daemon (29.6.1) with buildx 0.35.0 on the build
machine.

The CPU image was built and verified during design review on 2026-09-21, before
any plan was executed. Measured results: the build succeeds; `torch` is
`2.14.0+cpu`; zero `nvidia`/`cuda`/`triton` distributions are present; the
console script is on `PATH`; the container runs as uid/gid 1001; `/health`
returns `{"status":"ok",...}`; Docker reports the container `healthy`; the image
is **993 MB**. The server becomes ready **~7.1 s** after `docker run`, so
readiness must be polled rather than slept on.

**Dockerfile**

1. `docker build --build-arg TORCH_BACKEND=cpu -t von:cpu .` succeeds.
2. `docker run --rm von:cpu --help` prints `von serve` usage.
3. `docker run --rm --entrypoint python von:cpu -c "import torch; print(torch.__version__)"`
   prints `2.14.0+cpu`.
4. `docker run --rm --entrypoint python von:cpu -c "import importlib.metadata as m; print(sorted(d.metadata['Name'] for d in m.distributions() if d.metadata['Name'].lower().startswith(('nvidia','cuda','triton'))))"`
   prints `[]`.
5. `docker run -d -p 8000:8000 von:cpu`, then **poll** `GET /health` until it
   answers — startup is ~7.1 s, so a fixed `sleep` is a race — and confirm it
   returns `{"status":"ok",...}` and that Docker reports the container `healthy`.
6. `docker run --rm --entrypoint python von:cpu -c "import von"` succeeds.
7. Image size recorded for both variants.
8. `docker run --rm von:cpu serve --help` confirms the console script is on `PATH`.
9. Bandwidth permitting: pre-warm the volume and issue one real
   `POST /v1/systemone` request, asserting a well-formed response.

**Workflow**

10. `actionlint` (or equivalent YAML/schema validation) passes on
    `docker-publish.yml`.
11. The CalVer expression is evaluated in a shell to confirm the tag format
    matches `YYYY.MM.DD.HH.MM`.
12. The lowercasing expression is evaluated for the mixed-case repository name
    `Muhannadalrusayni/von` and confirmed to produce `muhannadalrusayni/von`.
13. The tag-generation shell block is executed locally against representative
    matrix values to confirm both variants emit exactly the tags in §9.

**Post-merge (requires a real publish)**

14. Confirm the workflow run succeeds and both packages appear in GHCR.
15. Confirm the CUDA variant publishes `:cuda-latest` and no bare `:cuda`.
16. `docker run --rm ghcr.io/astral-sh/uv:0.12.17 --version` prints `uv 0.12.17`,
    proving the pinned reference in the `Dockerfile` resolves.

The CUDA variant is **not** built locally during verification; it is not
runnable on the build machine and building it would pull ~8 GB. Its build path
differs from the CPU path only by the absence of the filter step, and is
exercised by CI.

## 14. Resolutions at spec review

Three items were raised and closed; nothing is outstanding.

1. **Tag scheme.** The word "edge" was removed from the scheme entirely — both
   the CalVer suffix and the floating pointer tags. The CUDA variant publishes
   `:cuda-latest` and no bare `:cuda`, and `:latest` resolves to the CPU build.
   The resulting tags are listed in §9, and the generation logic was executed
   locally to confirm the output (verification item 13).
2. **README section (D10).** Confirmed: a short "Container image" section will be
   added to `README.md`.
3. **uv toolchain (D11).** Raised after the first review, when the official uv
   Docker and PyTorch guides were checked against this design. The toolchain is
   now pinned, the export uses `--locked`, and the install runs under a BuildKit
   cache mount. The design was also validated by building the CPU image rather
   than by inspection alone; the measured results are in §13.
