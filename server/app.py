"""Compatibility server module required by OpenEnv multi-mode validator."""

from app import app as api_app
import uvicorn

app = api_app


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
