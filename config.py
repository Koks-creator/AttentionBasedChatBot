import logging
from pathlib import Path


class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent # maybe set static path sar

    # Folders
    MODEL_FOLDER: Path = ROOT_PATH / "models"

    # Model shit
    MODEL: str = "turn_model_extra_v2"

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = False
    WEB_APP_LOG_FILE: str = Path(ROOT_PATH) / "logs" / "web_app.log"
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_LOG_LEVEL: int = logging.DEBUG

    # LOGGER
    CLI_LOG_LEVEL: int = logging.INFO
    FILE_LOG_LEVEL: int = logging.INFO
    LOGS_PATH: Path = ROOT_PATH / "logs" / "logs.log"
