from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


POLICY_PATH = Path(__file__).parents[1] / "scripts" / "check_repo_policy.py"
SPEC = spec_from_file_location("check_repo_policy", POLICY_PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)
violation = POLICY.violation


def test_governance_files_are_allowed():
    assert violation("AGENTS.md") is None
    assert violation("CONTRIBUTING.md") is None


def test_upstream_assets_are_grandfathered():
    assert violation("README.md") is None
    assert violation("assets/logo.png") is None


def test_personal_content_is_rejected():
    assert violation("docs/notes.md") == "forbidden directory: docs/"
    assert violation("tmp/kernel.py") == "forbidden directory: tmp/"
    assert violation("recording.m4a") == "forbidden file type: .m4a"
    assert violation("weights/model.safetensors") == "forbidden file type: .safetensors"


def test_non_project_scripts_are_rejected():
    assert violation("scripts/transcribe_call.py") == "personal/interview helper script"
    assert violation("scripts/generate_review_pdf.py") == "document-generation script"
    assert violation("test.py") == "root scratch file"


def test_project_source_is_allowed():
    assert violation("nanovllm/engine/model_runner.py") is None
    assert violation("scripts/benchmark_baseline.py") is None
    assert violation("tests/test_sampler.py") is None
