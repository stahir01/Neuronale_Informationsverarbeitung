import pandas as pd
import numpy as np
import os

def import_dataset():
    current_directory = os.getcwd()
    relative_path = ['Dataset/movies.dat', 'Dataset/ratings.dat', 'Dataset/users.dat']

    for i in range(len(relative_path)):
        if i == 0:
            absolute_path = os.path.join(current_directory, relative_path[i])
            movies_dataset = pd.read_csv(absolute_path, sep='::', names=['movie_id', 'movie_name', 'movie_genre'], encoding='latin-1', engine='python')
        elif i == 1:
            absolute_path = os.path.join(current_directory, relative_path[i])
            rating_data = pd.read_csv(absolute_path, sep='::', names=['user_id', 'movie_id', 'movie_rating', 'time_period'], encoding='latin-1', engine='python')
        else:
            absolute_path = os.path.join(current_directory, relative_path[i])
            user_data = pd.read_csv(absolute_path, sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], encoding='latin-1', engine='python')

    
    movies_dataset['movie_genre'] = movies_dataset['movie_genre'].str.replace('|', ',')

    rating_data = (
    rating_data.groupby(['user_id', 'movie_id', 'movie_rating', 'time_period'])
    .size()
    .reset_index()
    .sort_values(['user_id', 'movie_id', 'movie_rating', 'time_period'], ascending=[True, True, True, True])
    .drop(columns=[0])
    )

    return movies_dataset, rating_data, user_data




if __name__ == '__main__':
    movies_dataset, rating_data, user_data = import_dataset()
    print(movies_dataset.head(10))
    print(rating_data.head(10))
    print(user_data.head(10))

