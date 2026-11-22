import os
from collections.abc import Generator
from sqlmodel import Session, SQLModel, create_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lab_user:lab_password@localhost:5432/hematology_db",
)

engine = create_engine(DATABASE_URL, echo=False)


def init_db() -> None:
    """Creates tables for all SQLModel models marked with table=True."""
    from app.schemas.clinical import PredictionRecord # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Provides a transactional database session per request."""
    with Session(engine) as session:
        yield session