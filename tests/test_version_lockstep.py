"""Every version site must agree with pyproject.toml (see scripts/check_release.py)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parent.parent / "scripts" / "check_release.py"
)
check_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_release)


def test_version_sites_in_lockstep() -> None:
    found = check_release.collect()
    assert check_release.problems(found, None) == [], found
