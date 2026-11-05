"""Write requirements.txt with the exact versions installed in the current environment.

Run it AFTER everything works (python -m pytest -q passes):
    python scripts/pin_requirements.py
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DIRECT = ["langgraph", "langchain-core", "langchain-google-genai",
          "langgraph-checkpoint-sqlite", "python-dotenv", "pytest"]

lines = [f"# Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
         " - exact versions of the direct dependencies used for the submitted runs"]
for name in DIRECT:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        sys.exit(f"{name} is not installed in this environment")
Path(__file__).resolve().parent.parent.joinpath("requirements.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
