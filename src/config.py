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


def _read_value(key: str, env_path: Path | None = None) -> str:
    """환경변수(우선)와 .env 에서 값 하나를 읽는다. 없으면 빈 문자열."""
    return os.environ.get(key) or _parse_env_file(env_path or ENV_PATH).get(key, "")


AWS_ENV_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
LANGFUSE_ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST")


def export_aws_credentials(env_path: Path | None = None) -> None:
    """`.env` 의 AWS·Langfuse 값을 환경변수로 내보낸다(boto3·langfuse 는 .env 를 읽지 않음). 이미 설정된 환경변수는 덮어쓰지 않는다."""
    file_values = _parse_env_file(env_path or ENV_PATH)
    for key in AWS_ENV_KEYS + LANGFUSE_ENV_KEYS:
        if file_values.get(key) and not os.environ.get(key):
            os.environ[key] = file_values[key]


def mockdata_use(env_path: Path | None = None) -> bool:
    """MOCKDATA_USE 가 False(또는 0/no/off)면 MCP 연동, 그 외(미설정 포함)는 True: LLM 추정만 쓰는 기존 방식."""
    return _read_value("MOCKDATA_USE", env_path).strip().lower() not in {"false", "0", "no", "off"}


def google_maps_api_key(env_path: Path | None = None) -> str:
    """Google Maps MCP 서버에 넘길 API 키를 읽는다. MOCKDATA_USE=False 인데 키가 없으면 ConfigError."""
    key = _read_value("GOOGLE_MAPS_API_KEY", env_path)
    if not key:
        raise ConfigError("GOOGLE_MAPS_API_KEY 가 설정되지 않았습니다 (.env 확인)")
    return key


DEFAULT_MYREALTRIP_URL = "https://mcp-servers.myrealtrip.com/mcp"


def myrealtrip_mcp_url(env_path: Path | None = None) -> str:
    """마이리얼트립 MCP(HTTP) 주소. MYREALTRIP_MCP_URL 로 바꿀 수 있고 없으면 공개 기본 주소를 쓴다(키 불필요)."""
    return _read_value("MYREALTRIP_MCP_URL", env_path) or DEFAULT_MYREALTRIP_URL


def openweather_api_key(env_path: Path | None = None) -> str:
    """OpenWeatherMap API 키를 읽는다(5일 예보용). 없으면 ConfigError. 환율(Frankfurter)은 키가 필요 없다."""
    key = _read_value("OPENWEATHER_API_KEY", env_path)
    if not key:
        raise ConfigError("OPENWEATHER_API_KEY 가 설정되지 않았습니다 (.env 확인)")
    return key


def naver_local_keys(env_path: Path | None = None) -> tuple[str, str]:
    """네이버 지역검색 API 키 한 쌍(ID, 키)을 읽는다. 하나라도 없으면 ConfigError."""
    key_id = _read_value("NAVER_LOCAL_API_KEY_ID", env_path)
    key = _read_value("NAVER_LOCAL_API_KEY", env_path)
    if not key_id or not key:
        raise ConfigError("NAVER_LOCAL_API_KEY_ID / NAVER_LOCAL_API_KEY 가 설정되지 않았습니다 (.env 확인)")
    return key_id, key


def naver_maps_keys(env_path: Path | None = None) -> tuple[str, str]:
    """네이버 Directions 15 API 키 한 쌍(ID, 키)을 읽는다. 하나라도 없으면 ConfigError."""
    key_id = _read_value("X_NCP_APIGW_API_KEY_ID", env_path)
    key = _read_value("X_NCP_APIGW_API_KEY", env_path)
    if not key_id or not key:
        raise ConfigError("X_NCP_APIGW_API_KEY_ID / X_NCP_APIGW_API_KEY 가 설정되지 않았습니다 (.env 확인)")
    return key_id, key
