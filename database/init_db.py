from database.connection import engine
from database.models import Base

# Import models so SQLAlchemy knows all tables.
from database import models


print("Creating database tables...")

Base.metadata.create_all(
    bind=engine
)

print("Database tables created successfully.")