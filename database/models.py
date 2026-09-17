from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
    Text
)

from sqlalchemy.sql import func

from .connection import Base


class Market(Base):
    __tablename__ = "markets"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String(100),
        unique=True,
        nullable=False
    )

    description = Column(
        String(255)
    )

    created_at = Column(
        DateTime,
        server_default=func.now()
    )


class HistoricalRecord(Base):
    __tablename__ = "historical_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    market_id = Column(
        Integer,
        ForeignKey("markets.id"),
        nullable=False
    )

    record_date = Column(
        Date,
        nullable=False
    )

    day_name = Column(
        String(20),
        nullable=False
    )

    data_status = Column(
        String(20),
        nullable=False,
        default="AVAILABLE"
    )

    col1 = Column(Integer)
    col2 = Column(Integer)
    col3 = Column(Integer)
    col4 = Column(Integer)
    col5 = Column(Integer)
    col6 = Column(Integer)
    col7 = Column(Integer)
    col8 = Column(Integer)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    prediction_date = Column(
        Date,
        nullable=False
    )

    target_column = Column(
        String(50),
        nullable=False
    )

    predicted_value = Column(
        Integer,
        nullable=False
    )

    ensemble_probability = Column(
        Float,
        nullable=False
    )

    confidence_score = Column(
        Float,
        nullable=False
    )

    confidence_category = Column(
        String(20),
        nullable=False
    )

    probability_margin = Column(
        Float,
        nullable=False
    )

    entropy = Column(
        Float,
        nullable=False
    )

    normalized_entropy = Column(
        Float,
        nullable=False
    )

    agreement_count = Column(
        Integer,
        nullable=False
    )

    total_models = Column(
        Integer,
        nullable=False
    )

    agreement_ratio = Column(
        Float,
        nullable=False
    )

    unique_predictions = Column(
        Integer,
        nullable=False
    )

    ensemble_probabilities = Column(
        Text,
        nullable=False
    )

    top_predictions = Column(
        Text,
        nullable=False
    )

    model_predictions = Column(
        Text,
        nullable=False
    )

    result_json = Column(
        Text,
        nullable=False
    )

    actual_value = Column(
        Integer,
        nullable=True
    )

    actual_record_date = Column(
        Date,
        nullable=True
    )

    prediction_status = Column(
        String(30),
        nullable=False,
        default="PENDING"
    )

    created_at = Column(
        DateTime,
        server_default=func.now()
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    market_id = Column(
        Integer,
        ForeignKey("markets.id"),
        nullable=False
    )

    prediction_date = Column(
        Date,
        nullable=False
    )

    predicted_value = Column(
        String(100)
    )

    model_name = Column(
        String(100)
    )

    confidence = Column(
        Float
    )

    created_at = Column(
        DateTime,
        server_default=func.now()
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    model_name = Column(
        String(100),
        nullable=False
    )

    training_date = Column(
        DateTime,
        server_default=func.now()
    )

    dataset_size = Column(
        Integer
    )

    accuracy = Column(
        Float
    )

    precision = Column(
        Float
    )

    recall = Column(
        Float
    )

    f1_score = Column(
        Float
    )

    status = Column(
        String(50)
    )


class PredictionEvaluation(Base):
    __tablename__ = "prediction_evaluations"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    prediction_id = Column(
        Integer,
        ForeignKey("predictions.id"),
        nullable=False
    )

    actual_value = Column(
        String(100)
    )

    is_correct = Column(
        Boolean
    )

    error_type = Column(
        String(255)
    )

    evaluated_at = Column(
        DateTime,
        server_default=func.now()
    )