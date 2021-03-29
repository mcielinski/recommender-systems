import pandas as pd
import numpy as np
from typing import Tuple


def get_grid(
    orders: pd.DataFrame,
    # columns: latitiude, longitude, vendor_id
    x_n_points: int,
    y_n_points: int
) -> pd.DataFrame:

    max_x = orders['longitude'].max()
    min_x = orders['longitude'].min()
    max_y = orders['latitude'].max()
    min_y = orders['latitude'].min()

    x_diff = (max_x - min_x) / x_n_points
    y_diff = (max_y - min_y) / y_n_points

    x_range = np.linspace(min_x, max_x, x_n_points)
    y_range = np.linspace(min_y, max_y, y_n_points)

    x = []
    y = []

    for i in x_range:
        for j in y_range:
            x.append(i)
            y.append(j)

    grid_points = pd.DataFrame({'longitude': x, 'latitude': y})
    grid_points = grid_points.reset_index().rename(columns={'index': 'id'})

    return grid_points, x_diff, y_diff


def find_nearest_point(
    location: Tuple[float],
    points: pd.DataFrame,
    # columns: longitude, latitude, id
    x_diff: float,
    y_diff: float
) -> int:

    x = location[0]
    y = location[1]

    points = points[
        (
            points['longitude'] < x + 2 * x_diff
        ) & (
            points['longitude'] > x - 2 * x_diff
        )
    ]
    points = points[
        (
            points['latitude'] < y + 2 * y_diff
        ) & (
            points['latitude'] > y - 2 * y_diff
        )
    ]

    points['distance'] = points.apply(
        lambda row: np.sqrt(
            (row['longitude'] - x)**2 + (row['latitude'] - y)**2
        ),
        axis=1
    )
    points = points.sort_values('distance')

    return points.iloc[0]['id']
