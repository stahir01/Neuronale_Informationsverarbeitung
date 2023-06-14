from sklearn.model_selection import train_test_split   
from data_prep import transform_data
from user_sequence import *
from data_split import tensor_batched_dataset
from transformer_model import create_model

if __name__ == '__main__':
    movies_data, rating_data, user_data = transform_data()
    # TODO: add all the elements to create working transformer model
    rating_sequence = prepare_user_sequence_data(ratings_data)
    # What is the idea behind the user_sequence ?