"""Compatibility server module required by OpenEnv multi-mode validator."""

from app import app as api_app
from app import main as api_main

app = api_app


def main() -> None:
    api_main()
