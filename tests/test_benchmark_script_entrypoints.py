from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "script",
    (
        "benchmark_qwen35_kernels.py",
        "benchmark_gptq_w4a16.py",
    ),
)
def test_benchmark_help_runs_outside_repository(script: str, tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
