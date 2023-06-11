import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split   
from data_prep import transform_data
from user_sequence import *
from data_split import tensor_batched_dataset


def create_input_model(sequence_size):


    #We just create input layers. It doesn't contain any data
    user_id_input = layers.input(input_shape=(1,), dtype=tf.string, name='user_id_input')

    # Define input layer for sequence of movie IDs
    sequence_movie_ids_input = layers.Input(
        name="sequence_movie_ids", shape=(sequence_size - 1), dtype=tf.string
    )

     # Define input layer for target movie ID
    target_movie_id_input = layers.Input(
        name="target_movie_id", shape=(1,), dtype=tf.string
    )

     # Define input layer for sequence of ratings
    sequence_ratings_input = layers.Input(
        name="sequence_ratings", shape=(sequence_size - 1), dtype=tf.int32
    )

     # Define input layer for target rating
    target_rating_input = layers.Input(
        name="target_rating", shape=(1,), dtype=tf.int32
    )

    return user_id_input, sequence_movie_ids_input, target_movie_id_input, sequence_ratings_input, target_rating_input



## Vocaubulary dataset for Embedding
def create_embedding_vocabulary(dataframe, col_list):
    
    vocabulary = {}

    #Create vocabulary for each column
    for col in col_list:
        unique_values = dataframe[col].unique().tolist()
        vocabulary[col] = unique_values
    
    return vocabulary










    

