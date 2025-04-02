import os
import random
from pathlib import Path

import modal
from dotenv import load_dotenv

APP_NAME = "sim"

# seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# paths
PARENT_PATH = Path(__file__).parent.parent
DB_SRC_PATH = PARENT_PATH / "db"
SRC_PATH = PARENT_PATH / "src"

# prompts
DEFAULT_USER_PROMPTS = [
    "keto and gum disease",
    "fasting and cognition",
    "CV health and exercise intensity",
]

# Modal
IN_PROD = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"
load_dotenv(".env" if IN_PROD else ".env.dev")
SECRETS = [
    modal.Secret.from_dotenv(
        path=PARENT_PATH, filename=".env" if IN_PROD else ".env.dev"
    )
]

MINUTES = 60  # seconds

PYTHON_VERSION = "3.12"


def naive_linechunk(toml_path):
    """Extract dependencies from pyproject.toml in a simple way."""
    content = Path(toml_path).read_text()
    deps_section = (
        content.split("[project]")[1].split("dependencies =")[1].split("]")[0]
    )
    deps = []
    for line in deps_section.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Remove quotes, commas, and brackets
            dep = line.replace('"', "").replace(",", "").replace("[", "").strip()
            if dep:
                deps.append(dep)
    return deps


PYPROJECT_DEPS = naive_linechunk(PARENT_PATH / "pyproject.toml")

IMAGE = (
    modal.Image.debian_slim(PYTHON_VERSION)
    .apt_install("git", "libpq-dev")  # add system dependencies
    .pip_install(*PYPROJECT_DEPS)  # add Python dependencies
    .add_local_file(PARENT_PATH / "favicon.ico", "/root/favicon.ico")
    .add_local_file(PARENT_PATH / "logo.png", "/root/logo.png")
    .add_local_file(PARENT_PATH / "pyproject.toml", "/root/pyproject.toml")
)
