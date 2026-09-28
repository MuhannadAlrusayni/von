#!/usr/bin/env python3
"""Assert every version site in the repo agrees. Exit 1 with a table if not.

Sites (all must derive from pyproject.toml's version X.Y.Z):
  pyproject.toml                 version = "X.Y.Z"
  src/von/__init__.py            __version__ = "X.Y.Z"
  js/package.json                "version": "X.Y.Z"
  src/von/engine.py              VON_VERSION = "X.Y"
  option_marker_backend.py       VON_MODEL_ID = "von-X.Y.0"
  js/src/client.ts               VON_MODEL = "von-X.Y.0"
  hf/MODEL_CARD.md               "# Von X.Y" and `von-sdk>=X.Y.Z`

Usage: scripts/check_release.py [--expect X.Y.Z]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _grep(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(pattern, text, re.M)
    return m.group(1) if m else "<missing>"


def collect() -> dict[str, str]:
    pkg = json.loads((ROOT / "js/package.json").read_text(encoding="utf-8"))
    return {
        "pyproject.toml": _grep("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "src/von/__init__.py": _grep("src/von/__init__.py", r'^__version__\s*=\s*"([^"]+)"'),
        "js/package.json": str(pkg.get("version", "<missing>")),
        "src/von/engine.py": _grep("src/von/engine.py", r'^VON_VERSION\s*=\s*"([^"]+)"'),
        "option_marker_backend.py": _grep(
            "src/von/backends/option_marker_backend.py", r'^VON_MODEL_ID\s*=\s*"([^"]+)"'
        ),
        "js/src/client.ts": _grep("js/src/client.ts", r'^export const VON_MODEL\s*=\s*"([^"]+)"'),
        "hf/MODEL_CARD.md#title": _grep("hf/MODEL_CARD.md", r"^# Von (\S+)"),
        "hf/MODEL_CARD.md#pip": _grep("hf/MODEL_CARD.md", r'von-sdk>=([0-9.]+)"'),
    }


def problems(found: dict[str, str], expect: str | None) -> list[str]:
    full = found["pyproject.toml"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", full):
        return [f"pyproject.toml version {full!r} is not X.Y.Z"]
    major_minor = full.rsplit(".", 1)[0]
    want = {
        "pyproject.toml": full,
        "src/von/__init__.py": full,
        "js/package.json": full,
        "src/von/engine.py": major_minor,
        "option_marker_backend.py": f"von-{major_minor}.0",
        "js/src/client.ts": f"von-{major_minor}.0",
        "hf/MODEL_CARD.md#title": major_minor,
        "hf/MODEL_CARD.md#pip": full,
    }
    out = [f"{k}: found {found[k]!r}, want {want[k]!r}" for k in want if found[k] != want[k]]
    if expect and full != expect:
        out.insert(0, f"requested release {expect!r} but pyproject.toml is {full!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", help="version the release is being cut for")
    args = ap.parse_args()
    found = collect()
    bad = problems(found, args.expect)
    width = max(len(k) for k in found)
    for k, v in found.items():
        print(f"  {k:<{width}}  {v}")
    if bad:
        print("\nversion lockstep FAILED:", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print(f"\nversion lockstep OK: {found['pyproject.toml']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
