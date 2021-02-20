# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import pandas as pd
import numpy as num
import mysql.connector
from sqlalchemy import create_engine
from typing import *
from pandas.core.api import DataFrame, Series
from sklearn.cluster import KMeans
from const import data_set_columns


def get_distance(x, input):
    dist = (x - input) ** 2
    dist = num.sum(dist)
    dist = num.sqrt(dist)
    return dist


def get_nearst_center_idx(centers, input):
    idx = 0
    prev_dist = get_distance(centers[0], input)
    for i in range(1, len(centers)):
        center = centers[i]
        curr_dist = get_distance(center, input)
        if prev_dist > curr_dist:
            idx = i
    return idx


class Clusterizer:

    def __init__(self, data_set: Union[DataFrame, Iterator[DataFrame]]):
        self._centers: DataFrame = None
        self.__kmeans: Union[KMeans, None] = None
        self.__data_set = data_set

    def clusterize(self, n_clusters: int = 3):
        self.__kmeans = KMeans(n_clusters=n_clusters)
        k = self.__kmeans
        k.fit(X=self.__data_set)

        centers = k.cluster_centers_
        centers = [num.concatenate(([i], center), axis=0) for i, center in enumerate(centers)]

        self._centers = pd.DataFrame(data=centers, columns=['id', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'])

    def assemble_sql_inserted_data(self, data_set: pd.DataFrame, original_data_frame: pd.DataFrame, d_type: str) -> \
            pd.DataFrame:

        if d_type != 'item' and d_type != 'place':
            raise ValueError(f'assemble_sql_inserted_data() '
                             f'd_type parameter should be item or place but your input is {d_type}')

        k = self.__kmeans
        predicted_centers = k.predict(data_set)

        data_with_cluster_idx = [
            [
                center,
                original_data_frame.iloc[i][f'{d_type}Id'],
                original_data_frame.iloc[i].p1,
                original_data_frame.iloc[i].p2,
                original_data_frame.iloc[i].p3,
                original_data_frame.iloc[i].p4,
                original_data_frame.iloc[i].p5,
                original_data_frame.iloc[i].p6
            ]
            for i, center in enumerate(predicted_centers)
        ]

        return pd.DataFrame(
            data=data_with_cluster_idx,
            columns=["clusterId", f"{d_type}Id"] + data_set_columns
        )
