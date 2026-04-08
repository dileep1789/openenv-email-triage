# Contributing to OpenEnv Email Triage

Thank you for your interest in contributing! 🚀

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Create a branch** for your feature or bug fix.
3. **Make your changes** in the new branch.
4. **Ensure tests pass** by running `python verify_env.py`.
5. **Submit a Pull Request** with a clear description of your changes.

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your tokens:
   - Copy `token.txt.template` to `token.txt`.
   - Add your `OPENAI_API_KEY`.

## Code Style

- Use Pydantic for data models.
- Maintain compatibility with the OpenEnv specification.
- Follow PEP 8 guidelines.
