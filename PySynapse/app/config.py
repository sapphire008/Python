"""Dynaconf settings for PySynapse."""
from pathlib import Path

from dynaconf import Dynaconf

ROOT = Path(__file__).resolve().parent.parent

settings = Dynaconf(
    envvar_prefix="PYSYNAPSE",
    settings_files=[
        str(ROOT / "settings.yaml"),
        str(ROOT / "settings.local.yaml"),
    ],
    environments=True,
)
