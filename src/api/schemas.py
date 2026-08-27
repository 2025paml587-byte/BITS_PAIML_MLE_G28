"""Request/response schemas for the trip-duration prediction API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    vendor_id: int = Field(..., description="Vendor identifier, e.g. 1 or 2")
    passenger_count: int = Field(..., ge=0, le=9)
    pickup_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: Literal["Y", "N"] = "N"
    algorithm: Optional[Literal["gradient_boosting", "linear_regression"]] = Field(
        default=None,
        description="Which trained model to use. Defaults to the configured default model.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vendor_id": 1,
                    "passenger_count": 1,
                    "pickup_datetime": "2016-03-14T09:30:00",
                    "pickup_longitude": -73.9855,
                    "pickup_latitude": 40.7580,
                    "dropoff_longitude": -73.9654,
                    "dropoff_latitude": 40.7829,
                    "store_and_fwd_flag": "N",
                    "algorithm": "gradient_boosting",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_trip_duration_seconds: float
    model_used: str
