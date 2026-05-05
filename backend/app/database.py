"""
MongoDB async connection using Motor.
Call `connect_db()` on startup and `close_db()` on shutdown.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

# Module-level references — populated at startup
client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    """Create the Motor client and select the database."""
    global client, db
    client = AsyncIOMotorClient(
        settings.MONGODB_URL,
        serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
        connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
        socketTimeoutMS=settings.MONGODB_TIMEOUT_MS,
    )
    db = client[settings.DATABASE_NAME]


async def close_db() -> None:
    """Gracefully close the Motor client."""
    global client
    if client is not None:
        client.close()


def get_db() -> AsyncIOMotorDatabase:
    """Return the current database handle (use inside route handlers)."""
    assert db is not None, "Database not initialised — call connect_db() first"
    return db
