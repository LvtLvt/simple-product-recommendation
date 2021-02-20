# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import pandas as pd
import numpy as num
import mysql.connector
from clusterizer import Clusterizer
from sqlalchemy import create_engine
from typing import *
from const import data_set_columns


cluster_table_name = '...Cluster'
cluster_map_table_name = '...ClusterMap'


def get_connection():
    # TODO set your driver info
    pass


def get_query(required_column: str) -> str:
    # TODO add your statistic query
    pass


def get_statistic_data_frame(statistic_type: str) -> pd.DataFrame:
    required_column = statistic_type + 'Id'
    data_frame = pd.read_sql(sql=get_query(required_column=required_column), con=connection,
                             columns=["id", required_column].extend(data_set_columns))

    if data_frame.empty:
        print(f'get_statistic_data_frame(): [{statistic_type}] data_frame not found, was db changed?')

    return data_frame


connection = get_connection()
tran = connection.begin_nested()
try:
    item_frame = get_statistic_data_frame(statistic_type='item')
    place_frame = get_statistic_data_frame(statistic_type='place')

    is_item_update_required = not item_frame.empty
    is_place_update_required = not place_frame.empty

    item_data_set: pd.DataFrame
    place_data_set = pd.DataFrame
    merged_data_set: pd.DataFrame = pd.DataFrame()

    if is_item_update_required:
        item_data_set = item_frame.drop(labels=['id', 'itemId'], axis=1)
        merged_data_set = merged_data_set.append(item_data_set)
    if is_place_update_required:
        place_data_set = place_frame.drop(labels=['id', 'placeId'], axis=1)
        merged_data_set = merged_data_set.append(place_data_set)

    if merged_data_set.empty:
        raise LookupError(f'data_set is empty, was db changed?')

    clusterizer = Clusterizer(data_set=merged_data_set)
    clusterizer.clusterize()
    centers = clusterizer._centers

    item_sql_inserted_data = clusterizer.assemble_sql_inserted_data(
        data_set=item_data_set, original_data_frame=item_frame, d_type='item')
    place_sql_inserted_data = clusterizer.assemble_sql_inserted_data(
        data_set=place_data_set, original_data_frame=place_frame, d_type='place')

    clusters_with_id = pd.DataFrame(data=centers, columns=["id"] + data_set_columns)
    print('-------------- clusters --------------')
    print(clusters_with_id)
    print('-------------- cluster item maps --------------')
    print(item_sql_inserted_data)
    print('-------------- cluster place maps --------------')
    print(place_sql_inserted_data)


    print('[DB] INSERT START...')
    connection.execute(f'DELETE FROM {cluster_map_table_name}')
    connection.execute(f'DELETE FROM {cluster_table_name}')
    clusters_with_id.to_sql(name=cluster_table_name, con=connection, if_exists='append', index=False)
    item_sql_inserted_data.to_sql(name=cluster_map_table_name, con=connection, if_exists='append', index=False)
    place_sql_inserted_data.to_sql(name=cluster_map_table_name, con=connection, if_exists='append', index=False)
    tran.commit()
    print('[DB] INSERT END...')

except mysql.connector.Error as error:
    print('database connection has not been established', error)
except Exception as error:
    tran.rollback()
    print('unexpected error', error)
finally:
    print('closed....')
    connection.close()
    exit(0)


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 \
                        + (points[i, 1] - curr_center[1]) ** 2 \
                        + (points[i, 2] - curr_center[2]) ** 2

        sse.append(curr_sse)
    return sse
