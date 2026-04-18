from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Single source of truth for all configuration.
    Every var is prefixed OS_ (Ocean Sentinel).
    """

    model_config = SettingsConfigDict(env_prefix="OS_", env_file=".env")

    mbari_bucket: str = "pacific-sound-16khz"
    mbari_sample_rate: int = 16_000
    mbari_segment_seconds: int = 60

    gfw_api_token: str = ""

    copernicus_username: str = ""
    copernicus_password: str = ""

    google_ai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    gemma_model: str = "gemma4:26b"

    sendgrid_api_key: str | None = None
    sendgrid_from_email: str = ""
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None

    spectrogram_n_mels: int = 128
    spectrogram_fmax: int = 1000
    correlation_radius_km: float = 50.0
    correlation_time_window_hours: int = 6

    database_url: str = "sqlite+aiosqlite:///data/ocean_sentinel.db"

    environment: str = "development"
    log_json: bool = False