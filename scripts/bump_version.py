#!/usr/bin/env python3
"""Rewrite every version site to X.Y.Z. Counterpart of check_release.py.

Usage: scripts/bump_version.py 1.3.2

Idempotent: re-running with the current version changes nothing. Always run
check_release.py afterwards; this script edits, it does not verify.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sub(path: str, pattern: str, repl: str) -> bool:
    p = ROOT / path
    old = p.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, old, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{path}: pattern {pattern!r} matched {n} times, expected 1")
    if new != old:
        p.write_text(new, encoding="utf-8")
    return new != old


def bump(full: str) -> list[str]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", full):
        sys.exit(f"version {full!r} is not X.Y.Z")
    mm = full.rsplit(".", 1)[0]
    model = f"von-{mm}.0"
    changed: list[str] = []
    edits = [
        ("pyproject.toml", r'^(version\s*=\s*)"[^"]+"', rf'\g<1>"{full}"'),
        ("src/von/__init__.py", r'^(__version__\s*=\s*)"[^"]+"', rf'\g<1>"{full}"'),
        ("src/von/engine.py", r'^(VON_VERSION\s*=\s*)"[^"]+"', rf'\g<1>"{mm}"'),
        ("src/von/backends/option_marker_backend.py", r'^(VON_MODEL_ID\s*=\s*)"[^"]+"', rf'\g<1>"{model}"'),
        ("js/src/client.ts", r'^(export const VON_MODEL\s*=\s*)"[^"]+"', rf'\g<1>"{model}"'),
        ("hf/MODEL_CARD.md", r"^# Von \S+", f"# Von {mm}"),
        ("hf/MODEL_CARD.md", r"^- \*\*Version:\*\* \S+ \(`von-[0-9.]+`", f"- **Version:** {mm} (`{model}`"),
        ("hf/MODEL_CARD.md", r'von-sdk>=[0-9.]+"', f'von-sdk>={full}"'),
        ("uv.lock", r'^(name = "von-sdk"\nversion = )"[^"]+"', rf'\g<1>"{full}"'),
    ]
    for path, pat, repl in edits:
        if _sub(path, pat, repl):
            changed.append(path)
    # package.json via regex too, to keep the file's formatting byte-identical.
    if _sub("js/package.json", r'^(\s*"version":\s*)"[^"]+"', rf'\g<1>"{full}"'):
        changed.append("js/package.json")
    pkg = json.loads((ROOT / "js/package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == full
    return sorted(set(changed))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for f in bump(sys.argv[1]):
        print(f"bumped {f}")
