import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import find_nearest_point, get_grid
from.model import Model


class Validator:

    def __init__(
        self,
        model: Model,
        orders: pd.DataFrame,
        # columns: latitiude, longitude, vendor_id
        vendors: pd.DataFrame,
        # columns: id, latitiude, longitude, vendor_rating
        grid_x_n_points: int,
        grid_y_n_points: int,
        test_size: float = 0.3,
        random_seed: int = 123
    ):
        tqdm.pandas()

        self.model = model

        # make grid
        print()
        print('Making grid...')
        self.grid_points, x_diff, y_diff = get_grid(
            orders,
            grid_x_n_points,
            grid_y_n_points
        )

        # find nearest points for orders
        print()
        print('Finding nearest aggregated points...')
        orders['point'] = orders.progress_apply(
            lambda row: find_nearest_point(
                (row['longitude'], row['latitude']),
                self.grid_points, x_diff, y_diff
            ),
            axis=1
        )

        # split points for test and train sets
        print()
        print()
        print('Spliting data...')
        unique_points = orders['point'].unique()
        np.random.seed(random_seed)
        self.test_points = np.random.choice(
            unique_points,
            size=int(len(unique_points) * test_size),
            replace=False
        )
        orders = orders[['point', 'vendor_id']]
        test_mask = orders['point'].isin(self.test_points)
        self.test_orders = orders[test_mask]
        train_orders = orders[~test_mask]

        # fit model
        print()
        print('Fitting model...')
        self.model.fit(
            train_orders,
            self.grid_points
        )

        print()
        print('Validator is ready!')

    def validate(
        self,
        n_recomendations: int
    ):

        n_relevant_items = 0
        n_recommended_items = 0
        n_possible_relevant_items = 0

        average_precisions = []

        for point in self.test_points:

            # prepare model input data
            point_info = self.grid_points[
                self.grid_points['id'] == point
            ]
            point_info = point_info.iloc[0]
            location = (
                point_info['longitude'],
                point_info['latitude']
            )

            # make recommendations
            recommended_vendors = self.model.predict(
                location,
                n_recomendations
            )

            # validate
            true_vendors = self.test_orders[
                self.test_orders['point'] == point
            ]['vendor_id'].unique()

            n_relevant_items += len(
                set(recommended_vendors).intersection(true_vendors)
            )
            n_recommended_items += len(recommended_vendors)
            n_possible_relevant_items += len(true_vendors)

            # average precision
            avg_prec = []
            for k in range(1, n_recomendations+1):
                k_recomendations = recommended_vendors[0:k]
                n_relevant_items_at_k = len(
                    set(k_recomendations).intersection(true_vendors)
                )
                if k_recomendations[-1] in true_vendors:
                    avg_prec.append(n_relevant_items_at_k / k)
            average_precisions.append(sum(avg_prec) / len(true_vendors))

        precision = n_relevant_items / n_recommended_items
        recall = n_relevant_items / n_possible_relevant_items
        mean_avg_prec = np.mean(np.array(average_precisions))

        return precision, recall, mean_avg_prec
