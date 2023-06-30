import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from data_prep import transform_data
from user_sequence import *
from embedding_vocabulary import *
from train_test_split import *
from csv_dataset_processing import *
from input_layer_and_embedding import *
import csv
from matplotlib import pyplot as plt

def create_model(
        user_features, 
        vocabulary, 
        movies_dataset, 
        genres,
        sequence_length=8, 
        hidden_units=[1024, 512, 256], 
        dropout_rate=0.1,
        num_heads=3,
        include_user_id=False, 
        include_user_features=False, 
        include_movie_features=False
        ):
     
    inputs = create_model_inputs(sequence_length)
    transformer_features, other_features = encode_input_features(
        inputs, user_features, vocabulary, movies_dataset, genres, sequence_length, include_user_id, include_user_features, include_movie_features
    )

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

other_approach = False

if __name__ == '__main__':
    movies_data, ratings_data, users_data, all_genres = transform_data()

    sequence_length = 8
    step_size = 4

    if other_approach:
        mixed_sequence = prepare_user_sequence_data_second_approach(ratings_data)
        rating_col_subsequence = append_subsequences_to_columns(mixed_sequence, ['unix_timestamp', 'movie_and_ratings'], sequence_length, step_size)
        rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_and_ratings'])
        rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 
        rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])
    else:
        rating_sequence = prepare_user_sequence_data(ratings_data)
        rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['unix_timestamp', 'movie_ids', 'ratings'], sequence_length, step_size)
        rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_ids', 'ratings'])
        
        rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 
        rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])

        rating_col_subsequence.rename(
        columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
        inplace=True,
        )

    print(rating_col_subsequence)

    train_data, test_data = split_data(rating_col_subsequence, 0.85)

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

    model = create_model(USER_FEATURES, vocab, movies_data, all_genres,sequence_length=sequence_length, hidden_units=[1024, 512, 256], dropout_rate=0.1, num_heads=3, include_user_id=False, include_user_features=True, include_movie_features=False)

    model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

    # Read the training data.
    train_dataset = get_dataset_from_csv("train_data.csv", CSV_HEADER, shuffle=True, batch_size=256)

    # Read the test data.
    test_dataset = get_dataset_from_csv("test_data.csv", CSV_HEADER, batch_size=265)

    # Fit the model with the training data.
    history = model.fit(train_dataset, epochs=25, validation_data=test_dataset)

    hist_df = pd.DataFrame(history.history) 

    # save to csv: 
    hist_csv_file = 'history_2nd.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Evaluate the model on the test data.
    x = model.evaluate(test_dataset, verbose=0, return_dict=True)
    # with open('dict.csv', 'w') as csv_file:  
    #     writer = csv.writer(csv_file)
    #     for key, value in x.items():
    #         writer.writerow([key, value])

    predict = model.predict(test_dataset)
    pd.DataFrame(predict, test_dataset).to_csv("predict.csv", index=False, sep="|", header=False)
