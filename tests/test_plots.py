import pandas as pd

from src.eda import plots


def make_engineered_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": [1, 2, 1, 2],
            "passenger_count": [1, 2, 1, 3],
            "store_and_fwd_flag": ["N", "Y", "N", "N"],
            "pickup_datetime": pd.to_datetime(
                ["2016-01-01 08:00", "2016-01-02 09:00", "2016-01-03 10:00", "2016-01-04 11:00"]
            ),
            "dropoff_datetime": pd.to_datetime(
                ["2016-01-01 08:10", "2016-01-02 09:20", "2016-01-03 10:15", "2016-01-04 11:30"]
            ),
            "pickup_latitude": [40.75, 40.76, 40.77, 40.78],
            "pickup_longitude": [-73.98, -73.97, -73.96, -73.95],
            "dropoff_latitude": [40.76, 40.77, 40.78, 40.79],
            "dropoff_longitude": [-73.97, -73.96, -73.95, -73.94],
            "trip_duration": [600, 1200, 900, 1800],
            "haversine_distance": [1.2, 2.4, 1.8, 3.1],
            "pickup_hour": [8, 9, 10, 11],
            "pickup_day_of_week": [4, 5, 6, 0],
            "pickup_month": [1, 1, 1, 1],
        }
    )


def test_all_plot_functions_write_their_output_file(tmp_path):
    df = make_engineered_df()

    written = [
        plots.plot_passenger_count(df, tmp_path),
        plots.plot_categorical_distributions(df, tmp_path),
        plots.plot_pickup_dropoff_map(df, tmp_path, sample_size=2),
        plots.plot_haversine_distance_distribution(df, tmp_path),
        plots.plot_distance_vs_duration(df, tmp_path, sample_size=2),
        plots.plot_pickup_distribution_by_time(df, tmp_path),
        plots.plot_duration_by_time(df, tmp_path),
    ]

    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0
