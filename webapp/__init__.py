import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from turbo_chat_bot import ChatBot

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

def setup_logging(app: Flask) -> None:
    """Configure logging for the application"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(Config.WEB_APP_LOG_FILE)
    file_handler.setLevel(Config.FILE_LOG_LEVEL)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(Config.FILE_LOG_LEVEL)
    app.logger.propagate = False  # Prevent duplicate logs


def create_app() -> tuple[Flask]:
    """Initialize and configure the Flask application"""
    app = Flask(
        "ChatBotApp",
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR
    )
    
    setup_logging(app)
    app.logger.info("Starting web app")
    
    app.secret_key = os.getenv("SECRET_KEY")
    
    return app

app = create_app()

models = {
    "model_3_turns": ChatBot(
        model_path=Config.MODEL_FOLDER / "turn_model_extra_v2" / "best_turowy_v2.weights.h5",
        vocab_path=Config.MODEL_FOLDER / "turn_model_extra_v2" / "vocab_turowy_v2.txt",
        params_path=Config.MODEL_FOLDER / "turn_model_extra_v2" / "params_turowy_v2.json"
    ),
    "model_4_turns": ChatBot(
        model_path=Config.MODEL_FOLDER / "turn_model_extra_v3" / "best.weights.h5",
        vocab_path=Config.MODEL_FOLDER / "turn_model_extra_v3" / "vocab_turowy_v2.txt",
        params_path=Config.MODEL_FOLDER / "turn_model_extra_v3" / "params.json"
    )
}
from webapp import routes

app.logger.info("App started")