import pandas as pd
import numpy as np
from data_prep import * 
from user_sequence import *


def split_data(dataframe, train_size):
    train_length = int(len(dataframe.index) * train_size)
    train_data = dataframe[:train_length]
    test_data = dataframe[train_length:]

    return train_data, test_data


if __name__ =='__main__':
    movies_data, ratings_data, users_data, all_genres = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)

    sequence_length = 8
    step_size = 4
    rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['unix_timestamp', 'movie_ids', 'ratings'], sequence_length, step_size)

    rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_ids', 'ratings'])
    rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 
    rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])

    rating_col_subsequence.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
    )

    train_data, test_data = split_data(rating_col_subsequence, 0.85)

    train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
    test_data.to_csv("test_data.csv", index=False, sep="|", header=False)