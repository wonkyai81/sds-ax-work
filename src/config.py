"""환경 설정 로더: Bedrock 리전·모델 ID를 .env 와 환경변수에서 읽는다."""
import os
from dataclasses import dataclass
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
REQUIRED_KEYS = ("AWS_DEFAULT_REGION", "BEDROCK_MODEL_ID")


class ConfigError(RuntimeError):
    """필수 설정이 없을 때 발생하는 오류."""


@dataclass(frozen=True)
class Settings:
    """Bedrock 접속 설정 값 묶음."""

    region: str
    model_id: str


def _parse_env_file(path: Path) -> dict[str, str]:
    """`.env` 파일을 KEY=VALUE 사전으로 읽는다. 파일이 없으면 빈 사전."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_settings(env_path: Path | None = None) -> Settings:
    """환경변수(우선)와 .env 에서 리전·모델 ID를 읽어 Settings 로 돌려준다.

    값이 없으면 ConfigError 를 낸다. 코드에는 기본값을 두지 않으며 사용자 키 입력은 받지 않는다.
    """
    file_values = _parse_env_file(env_path or ENV_PATH)
    found: dict[str, str] = {}
    for key in REQUIRED_KEYS:
        value = os.environ.get(key) or file_values.get(key)
        if not value:
            raise ConfigError(f"{key} 가 설정되지 않았습니다 (.env 확인)")
        found[key] = value
    return Settings(region=found["AWS_DEFAULT_REGION"], model_id=found["BEDROCK_MODEL_ID"])
