"""AC5: 설정 로더 테스트 (.env/환경변수에서 읽고, 코드에 값이 없고, 키 입력이 없음)."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """테스트 간 환경변수를 비운다."""
    for key in config.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_reads_from_env_file(tmp_path):
    """.env 파일에서 리전과 모델 ID를 읽는다."""
    env = tmp_path / ".env"
    env.write_text("AWS_DEFAULT_REGION=ap-test-9\n# c\nBEDROCK_MODEL_ID='model-x'\n", encoding="utf-8")
    s = config.load_settings(env)
    assert (s.region, s.model_id) == ("ap-test-9", "model-x")


def test_env_var_overrides_file(tmp_path, monkeypatch):
    """환경변수가 .env 보다 우선한다."""
    env = tmp_path / ".env"
    env.write_text("AWS_DEFAULT_REGION=a\nBEDROCK_MODEL_ID=b\n", encoding="utf-8")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "r1")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "m1")
    s = config.load_settings(env)
    assert (s.region, s.model_id) == ("r1", "m1")


@pytest.mark.parametrize("missing", config.REQUIRED_KEYS)
def test_missing_value_raises(tmp_path, monkeypatch, missing):
    """값이 없으면 기본값 없이 오류를 낸다."""
    for key in config.REQUIRED_KEYS:
        if key != missing:
            monkeypatch.setenv(key, "v")
    with pytest.raises(config.ConfigError):
        config.load_settings(tmp_path / "none.env")


def test_no_hardcoded_values_or_user_keys():
    """소스에 리전·모델 ID 값이 없고 API 키 입력 항목이 없다."""
    src = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    assert not re.search(r"us-east-1|anthropic\.claude", src)
    assert not re.search(r"ACCESS_KEY|SECRET|api_key", src, re.I)
    assert set(config.Settings.__dataclass_fields__) == {"region", "model_id"}
