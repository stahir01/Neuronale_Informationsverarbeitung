import pandas as pd
from data_prep import transform_data
from user_sequence import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv

"""
Idea:
1) Divide data into train and test
2) Create an array that takes last 7 movies and predicts the last one
3) Same for ratings
4) Then we divide it these rating by user
"""

def process_features(features):
    movie_ids_string = features["sequence_movie_ids"]
    sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

    # The last movie id in the sequence is the target movie.
    features["target_movie_id"] = sequence_movie_ids[:, -1]
    features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

    ratings_string = features["sequence_ratings"]
    sequence_ratings = tf.strings.to_number(
        tf.strings.split(ratings_string, ","), tf.dtypes.float32
    ).to_tensor()

    # The last rating in the sequence is the target for the model to predict.
    target = sequence_ratings[:, -1]
    features["sequence_ratings"] = sequence_ratings[:, :-1]

    return features, target

def get_dataset_from_csv(csv_file_path, column_name, shuffle=False, batch_size=128, epoch=1):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=column_name,
        num_epochs=epoch,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process_features)

    return dataset

def get_top_k(csv_file_path, k):
    real_ratings = []
    real_movies = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            real_ratings.append(row[2].split(",")[-1])
            real_movies.append(row[1].split(",")[-1])
    
    top_k_ind = np.argpartition(np.array(real_ratings), -k)[-k:]
    top_k = np.array(real_movies)[top_k_ind]
    return top_k

print(get_top_k('test_data.csv', 10))
"""
#Convert pandas dataset into tensor
def tensor_batched_dataset(train_dataset, batch_size, number=7):

    user_ids = []
    movie_sequences = []
    movie_predictions = []
    rating_sequences = []
    rating_predictions = []

    #Convert pandas dataset into list
    for index, row in train_dataset.iterrows():
        user_ids.append(str(index))
        movies = [movie.strip() for movie in row['movies_id'].split(',')[:number]]         #First number movies
        movie_pred = row['movies_id'].split(',')[-1].strip()                          #Last movie as prediction
        ratings = [int(rating) for rating in row['movie_ratings'].split(',')[:number]]    #Convert ratings to int
        rating_pred = int(row['movie_ratings'].split(',')[-1])
        movie_sequences.append(movies)                                              #Shape: (len(dataset), number)
        rating_sequences.append(ratings)                                            
        movie_predictions.append([movie_pred])                                      #Shape: (len(dataset), 1)
        rating_predictions.append([rating_pred])                                    
    
    

    #print(f"Input shape: {(len(movie_sequences), len(movie_sequences[0]))}, Prediction shape: {(len(movie_predictions), len(movie_predictions[0]))}")

    #Convert list into tensor
    user_ids_tensor = tf.convert_to_tensor(user_ids, dtype=tf.string)
    movie_sequences_tensor = tf.convert_to_tensor(movie_sequences, dtype=tf.string)
    movie_predictions_tensor = tf.convert_to_tensor(movie_predictions, dtype=tf.string)
    rating_sequences_tensor = tf.convert_to_tensor(rating_sequences, dtype=tf.int32)
    rating_predictions_tensor = tf.convert_to_tensor(rating_predictions, dtype=tf.int32)

    tensor_dataset = tf.data.Dataset.from_tensor_slices((user_ids_tensor, movie_sequences_tensor, movie_predictions_tensor, rating_sequences_tensor, rating_predictions_tensor))

    #Shuffle and batch dataset   
    dataset = tensor_dataset.shuffle(buffer_size=len(movie_sequences)).batch(batch_size)

    return dataset

if __name__ == '__main__':
    movies_data, ratings_data, users_data = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)

    rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['time_period', 'movies_id', 'movie_ratings'], 8, 4)    
    rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['time_period', 'movies_id', 'movie_ratings'])
    rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 


    train_dataset, test_dataset = train_test_split(rating_col_subsequence, test_size=0.2, random_state=42, shuffle=True)   
    print(f"Length of Actual Dataset: {len(rating_col_subsequence)}, Length of Train Dataset: {len(train_dataset)}, Length of Test Dataset: {len(test_dataset)}")

    tensor_data = tensor_batched_dataset(train_dataset, 32)

    i = 0
    for batch in tensor_data:
        while i < 2:
            user_ids, movie_sequences, movie_predictions, rating_sequences, rating_predictions = batch
            
             # Print the batch elements
            print("User IDs:")
            print(user_ids)
            print("Movie Sequences:")
            print(movie_sequences)
            print("Movie Predictions:")
            print(movie_predictions)
            print("Rating Sequences:")
            print(rating_sequences)
            print("Rating Predictions:")
            print(rating_predictions)
            print("-----------------------------")
            i += 1
""" 

'''

What we need to do:
- create ordered lists
- create sequences 
- save these sequences seperately
- add users info/data to time/rating infos
- split into test and training sequences
Transformer architecture:
- Encoder
- create model
- run training and evaluation

Questions:
- 2 sequences: by movie_rating and by watch_order
- movie_rating should enclude user infos
- How to encode?
- time period?


'''


