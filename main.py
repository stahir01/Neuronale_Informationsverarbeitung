import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from data_prep import transform_data
from user_sequence import *
from embedding_vocabulary import *
from train_test_split import *
from csv_dataset_processing import *
from input_layer_and_embedding import *
from transformer_model import *


def run(sequence_len=8, step_siz=4, data_split=0.85, hidden_unit=[256, 128], dropout__rate=0.1, num__heads=3, learning__rate=0.01, batch__size=256, total_epochs=2):

    """
    This function runs the whole process of the model. It takes the parameters for the model and the training process as input.

    Args:
        sequence_len (int, optional): Length of the sequence. Defaults to 8.
        step_siz (int, optional): Step size of the sequence. Defaults to 4.
        data_split (float, optional): Split of the data into train and test data. Defaults to 0.85.
        hidden_unit (list, optional): List of hidden units for the model. Defaults to [256, 128].
        dropout__rate (float, optional): Dropout rate for the model. Defaults to 0.1.
        num__heads (int, optional): Number of heads for the model. Defaults to 3.
        learning__rate (float, optional): Learning rate for the model. Defaults to 0.01.
        batch__size (int, optional): Batch size for the model. Defaults to 256.
        total_epochs (int, optional): Number of epochs for the model. Defaults to 2.

    Returns:
        None
    
    """


    movies_data, ratings_data, users_data, all_genres = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)

    sequence_length = sequence_len
    step_size = step_siz
    rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['unix_timestamp', 'movie_ids', 'ratings'], sequence_length, step_size)

    rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_ids', 'ratings'])
    rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 
    rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])

    rating_col_subsequence.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
    )

    print(rating_col_subsequence)

    train_data, test_data = split_data(rating_col_subsequence, data_split)

    if os.path.exists("train_data.csv"):
        os.remove("train_data.csv")
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")
    
    train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
    test_data.to_csv("test_data.csv", index=False, sep="|", header=False)


    vocab = create_embedding_vocabulary([movies_data, ratings_data, users_data], ['user_id', 'movie_id', 'sex', 'age_group', 'occupation'])

    USER_FEATURES = ["sex", "age_group", "occupation"]
    MOVIE_FEATURES = ["genres"]
    CSV_HEADER = list(rating_col_subsequence.columns)

    model = create_model(USER_FEATURES, vocab, movies_data, all_genres,sequence_length=sequence_length, hidden_units=hidden_unit, dropout_rate=dropout__rate, num_heads=num__heads, include_user_id=False, include_user_features=False, include_movie_features=False)


    model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=learning__rate),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

    # Read the training data.
    train_dataset = get_dataset_from_csv("train_data.csv", CSV_HEADER, shuffle=True, batch_size=batch__size)

    # Fit the model with the training data.
    model.fit(train_dataset, epochs=total_epochs)

    # Read the test data.
    test_dataset = get_dataset_from_csv("test_data.csv", CSV_HEADER, batch_size=batch__size)

    # Evaluate the model on the test data.
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MAE: {round(rmse, 3)}")

if __name__ == '__main__':
    run()

    