"""Reject repository content that violates the source-only policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


ALLOWED_NON_SOURCE_FILES = {
    ".gitignore",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "assets/logo.png",
    "pyproject.toml",
}

FORBIDDEN_DIRECTORIES = {
    "benchmark_results",
    "docs",
    "offline_bundle",
    "output",
    "tmp",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".aac",
    ".avi",
    ".bin",
    ".doc",
    ".docx",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".md",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".onnx",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".rtf",
    ".safetensors",
    ".tar",
    ".tgz",
    ".txt",
    ".wav",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}

FORBIDDEN_SCRIPT_MARKERS = (
    "interview",
    "transcribe_",
    "generate_resume_",
    "build_tencent_",
    "update_interview_",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def violation(path_string: str) -> str | None:
    if path_string in ALLOWED_NON_SOURCE_FILES:
        return None
    path = PurePosixPath(path_string)
    if path.parts and path.parts[0] in FORBIDDEN_DIRECTORIES:
        return f"forbidden directory: {path.parts[0]}/"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"forbidden file type: {path.suffix.lower()}"
    if path.parts and path.parts[0] == "scripts":
        lowered = path.name.lower()
        if "generate_" in lowered and "_pdf" in lowered:
            return "document-generation script"
        if any(marker in lowered for marker in FORBIDDEN_SCRIPT_MARKERS):
            return "personal/interview helper script"
    if path_string == "test.py":
        return "root scratch file"
    return None


def main() -> int:
    violations = [
        (path, reason)
        for path in tracked_files()
        if (reason := violation(path)) is not None
    ]
    if violations:
        print("Repository policy violations:", file=sys.stderr)
        for path, reason in violations:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    print("Repository policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
