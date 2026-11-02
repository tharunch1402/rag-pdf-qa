import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    APP_NAME = 'Production RAG'
    APP_VERSION = '0.0.1'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

settings = Settings()