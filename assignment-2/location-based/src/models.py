from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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


class RandomModel(Model):
    # Returns random recommendations

    def fit(
        self,
        train_orders: pd.DataFrame,
        # columns: point, vendor_id
        points: pd.DataFrame
        # columns: id, x, y
    ) -> None:
        self.is_fitted = True

    def get_ranking(
        self,
        location: Tuple[float, float]
    ) -> List[int]:

        self.vendors.sample(
            frac=1,
            random_state=self.random_seed
        )

        return list(self.vendors['id'])


class DistanceModel(Model):
    # Recommends the nearest vendors

    def fit(
        self,
        train_orders: pd.DataFrame,
        # columns: point, vendor_id
        points: pd.DataFrame
        # columns: id, x, y
    ) -> None:
        self.is_fitted = True

    def get_ranking(
        self,
        location: Tuple[float, float]
    ) -> List[int]:

        x = location[0]
        y = location[1]

        self.vendors['distance'] = self.vendors.apply(
            lambda row: np.sqrt(
                (row['longitude'] - x)**2 + (row['latitude'] - y)**2
            ),
            axis=1
        )

        self.vendors = self.vendors.sort_values('distance')

        return list(self.vendors['id'])


class ClusterModel(Model):
    # Recommends vendors from the nearest cluster with the highest ratings

    def __init__(
        self,
        n_clusters: int,
        vendors: pd.DataFrame,
        # columns: id, latitiude, longitude, vendor_rating
        random_seed: int = 123,
    ) -> None:
        super().__init__(
            vendors,
            random_seed
        )
        self.n_clusters = n_clusters

    def fit(
        self,
        train_orders: pd.DataFrame,
        # columns: point, vendor_id
        points: pd.DataFrame
        # columns: id, x, y
    ) -> None:
        self.is_fitted = True

        kmeans = KMeans(self.n_clusters)
        self.vendors['cluster'] = kmeans.fit_predict(
            self.vendors[['latitude', 'longitude']]
        )

        # having knowledge from data exploration we will fix
        # cluster of one specific vendor if n_clusters == 3
        if self.n_clusters == 3:
            row_to_fix = self.vendors[
                self.vendors['id'] == 845
            ].index[0]
            to_cluster = self.vendors[
                self.vendors['id'] == 78
            ].iloc[0]['cluster']
            self.vendors.at[row_to_fix, 'cluster'] = to_cluster

    def get_ranking(
        self,
        location: Tuple[float, float]
    ) -> List[int]:

        x = location[0]
        y = location[1]

        # compute distances to vendors
        self.vendors['distance'] = self.vendors.apply(
            lambda row: np.sqrt(
                (row['longitude'] - x)**2 + (row['latitude'] - y)**2
            ),
            axis=1
        )
        self.vendors = self.vendors.sort_values('distance')

        # get cluster of the nearest vendor
        nearest_cluster = self.vendors.iloc[0]['cluster']

        # get vendors from that cluster and sort by rating
        sub_vendors = self.vendors[self.vendors['cluster'] == nearest_cluster]
        sub_vendors = self.vendors.sort_values(
            'vendor_rating',
            ascending=False
        )

        return list(sub_vendors['id'])


class GravityModel(Model):
    # Recommends vendors using gravity model with vendors' ratings as mass

    def fit(
        self,
        train_orders: pd.DataFrame,
        # columns: point, vendor_id
        points: pd.DataFrame
        # columns: id, x, y
    ) -> None:
        self.is_fitted = True

    def get_ranking(
        self,
        location: Tuple[float, float]
    ) -> List[int]:

        x = location[0]
        y = location[1]

        # compute distances to vendors
        self.vendors['distance'] = self.vendors.apply(
            lambda row: np.sqrt(
                (row['longitude'] - x)**2 + (row['latitude'] - y)**2
            ),
            axis=1
        )
        self.vendors = self.vendors.sort_values('distance')

        # fill mising ratings with mean
        self.vendors['vendor_rating'].fillna(
            (self.vendors['vendor_rating'].mean()),
            inplace=True
        )

        # compute gravity rank
        self.vendors['gravity'] = \
            self.vendors['vendor_rating'] / self.vendors['distance']

        # sort by gravity rank
        self.vendors = self.vendors.sort_values(
            'gravity',
            ascending=False
        )

        return list(self.vendors['id'])
