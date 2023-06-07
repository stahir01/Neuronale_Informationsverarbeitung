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

    return movies_dataset, rating_data, user_data


#Convert genre column to binary indicators
def apply_genre_indicators(movies_dataset): 
    movies_dataset['movie_genre'] = movies_dataset['movie_genre'].str.split('|')
    distinct_genres = movies_dataset['movie_genre'].explode().unique()
    for genre in distinct_genres:
        movies_dataset[genre] = movies_dataset['movie_genre'].apply(lambda values: int(genre in values))
    movies_dataset = movies_dataset.drop(columns=['movie_genre'])
    return movies_dataset



def transform_data():
    movies_dataset, rating_data, user_data = import_dataset()

    movies_dataset['movie_id'] = movies_dataset['movie_id'].apply(lambda x: f"movie_{x}")
    rating_data['user_id'] = rating_data['user_id'].apply(lambda x: f"user_{x}")
    rating_data['movie_id'] = rating_data['movie_id'].apply(lambda x: f"movie_{x}")
    user_data['user_id'] = user_data['user_id'].apply(lambda x: f"user_{x}")
    user_data['age'] = user_data['age'].apply(lambda x: f"age_group_{x}")
    user_data['occupation'] = user_data['occupation'].apply(lambda x: f"occupation_{x}")

    movies_dataset = apply_genre_indicators(movies_dataset)

    rating_data = (
        rating_data.groupby(['user_id', 'movie_id', 'movie_rating', 'time_period'])
        .size()
        .reset_index()
        .sort_values(['user_id', 'movie_id', 'movie_rating', 'time_period'], ascending=[True, True, True, True])
        .drop(columns=[0])
    )

    return movies_dataset, rating_data, user_data


if __name__ == '__main__':
    movies_data, ratings_data, users_data = transform_data()
    print(movies_data.head(10))


