import os

from csv_dataset_processing import get_dataset_from_csv
from embedding_vocabulary import create_embedding_vocabulary
from train_test_split import split_data
from transformer_model import create_model
from tensorflow import keras
from user_sequence import *


if __name__ == '__main__':

    print('Preparing data')
    movies_data, ratings_data, users_data, all_genres = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)

    print('Creating sequences')
    sequence_length = 8
    step_size = 4
    rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['unix_timestamp', 'movie_ids', 'ratings'],
                                                            sequence_length, step_size)

    rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_ids', 'ratings'])
    rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"),
                                                         on="user_id")  # Combine with user data
    rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])

    rating_col_subsequence.rename(
        columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
        inplace=True,
    )

    print(rating_col_subsequence)

    print('Splitting data in train and test sets')
    train_data, test_data = split_data(rating_col_subsequence, 0.85)

    if os.path.exists("train_data.csv"):
        os.remove("train_data.csv")
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")

    train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
    test_data.to_csv("test_data.csv", index=False, sep="|", header=False)

    print('Creating model inputs')
    vocab = create_embedding_vocabulary([movies_data, ratings_data, users_data],
                                        ['user_id', 'movie_id', 'sex', 'age_group', 'occupation'])

    USER_FEATURES = ["sex", "age_group", "occupation"]
    MOVIE_FEATURES = ["genres"]
    CSV_HEADER = list(rating_col_subsequence.columns)

    print('Creating and running the model')
    model = create_model(USER_FEATURES, vocab, movies_data, all_genres, sequence_length=sequence_length,
                         hidden_units=[256, 128], dropout_rate=0.1, num_heads=3, include_user_id=False,
                         include_user_features=False, include_movie_features=False)

    model.compile(
        optimizer=keras.optimizers.legacy.Adagrad(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # Read the training data.
    train_dataset = get_dataset_from_csv("train_data.csv", CSV_HEADER, shuffle=True, batch_size=256)

    # Fit the model with the training data.
    model.fit(train_dataset, epochs=5)

    # Read the test data.
    test_dataset = get_dataset_from_csv("test_data.csv", CSV_HEADER, batch_size=265)

    # Evaluate the model on the test data.
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MAE: {round(rmse, 3)}")