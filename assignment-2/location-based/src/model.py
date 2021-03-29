from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class Model(ABC):

    def __init__(
        self,
        vendors: pd.DataFrame,
        # columns: id, latitiude, longitude, vendor_rating
        random_seed: int = 123
    ) -> None:
        self.vendors = vendors
        self.random_seed = random_seed
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_orders: pd.DataFrame,
        # columns: point, vendor_id
        points: pd.DataFrame
        # columns: id, x, y
    ) -> None:
        pass

    @abstractmethod
    def get_ranking(
        self,
        location: Tuple[float, float]
    ) -> List[int]:
        pass

    def predict(
        self,
        location,
        n_recomendations
    ) -> List[int]:

        if not self.is_fitted:
            raise Exception('First fit the model!')

        ranking = self.get_ranking(location)
        return ranking[0:n_recomendations]
