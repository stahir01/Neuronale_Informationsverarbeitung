import pandas as pd
import numpy as np
import os

def import_dataset():
    current_directory = os.getcwd()
    relative_path = ['Dataset/movies.dat', 'Dataset/ratings.dat', 'Dataset/users.dat']

    for i in range(len(relative_path)):
        if i == 0:
            absolute_path = os.path.join(current_directory, relative_path[i])
            movies_dataset = pd.read_csv(absolute_path, sep='::', names=['movie_id', 'title', 'genres'], encoding='latin-1', engine='python')
        elif i == 1:
            absolute_path = os.path.join(current_directory, relative_path[i])
            rating_data = pd.read_csv(absolute_path, sep='::', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'], encoding='latin-1', engine='python')
        else:
            absolute_path = os.path.join(current_directory, relative_path[i])
            user_data = pd.read_csv(absolute_path, sep='::', names=['user_id', 'sex', 'age_group', 'occupation', 'zip_code'], encoding='latin-1', engine='python')

    return movies_dataset, rating_data, user_data


#Convert genre column to binary indicators
def apply_genre_indicators(movies_dataset):
    movies_dataset['genres'] = movies_dataset['genres'].str.split('|')
    distinct_genres = movies_dataset['genres'].explode().unique()
    all_genres = list(distinct_genres)

    for genre in distinct_genres:
        movies_dataset[genre] = movies_dataset['genres'].apply(lambda values: int(genre in values))

    movies_dataset['genres'] = movies_dataset['genres'].apply('|'.join)  # Join genres using '|'
    #movies_dataset = movies_dataset.drop(columns=['movie_genre'])
    return movies_dataset, all_genres



def transform_data():
    movies_data, ratings_data, user_data = import_dataset()

    movies_data['movie_id'] = movies_data['movie_id'].apply(lambda x: f"movie_{x}")
    ratings_data['user_id'] = ratings_data['user_id'].apply(lambda x: f"user_{x}")
    ratings_data['movie_id'] = ratings_data['movie_id'].apply(lambda x: f"movie_{x}")
    user_data['user_id'] = user_data['user_id'].apply(lambda x: f"user_{x}")
    user_data['age_group'] = user_data['age_group'].apply(lambda x: f"age_group_{x}")
    user_data['occupation'] = user_data['occupation'].apply(lambda x: f"occupation_{x}")

    movies_dataset, all_genres = apply_genre_indicators(movies_data)

    rating_data = (
        ratings_data.groupby(['user_id', 'unix_timestamp', 'movie_id',  'rating'])  #moved time_period to second column for sequence creation based on time_period
        .size()
        .reset_index()
        .sort_values(['user_id', 'unix_timestamp', 'movie_id', 'rating'], ascending=[True, True, True, True])
        .drop(columns=[0])
    )

    return movies_dataset, rating_data, user_data, all_genres


if __name__ == '__main__':
    movies_data, rating_data, user_data, all_genres = transform_data()
    print(movies_data.head(10))


