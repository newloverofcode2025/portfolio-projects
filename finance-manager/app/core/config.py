from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "AI-Powered Finance Manager"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    # Database
    DATABASE_URL: str
    
    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Settings
    MODEL_UPDATE_FREQUENCY: int = 24  # hours
    PREDICTION_THRESHOLD: float = 0.8

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
