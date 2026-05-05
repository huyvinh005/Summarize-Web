"""
Application configuration.
Loads settings from environment variables or .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb+srv://vinhpro_db_user:Vinhpro123@cluster0.kl7vhle.mongodb.net/ai_summarizer?retryWrites=true&w=majority"
    DATABASE_NAME: str = "ai_summarizer"
    MONGODB_TIMEOUT_MS: int = 5000

    # JWT
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # AI (placeholder — replace with your own model config)
    AI_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"

    # Frontend
    FRONTEND_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @property
    def frontend_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.FRONTEND_ORIGINS.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"


settings = Settings()
