import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split   
from data_prep import transform_data
from user_sequence import *
from data_split import tensor_batched_dataset
from tensorflow.keras.layers import StringLookup

USER_FEATURES = []

def create_input_model(sequence_size=4):


    #We just create input layers. It doesn't contain any data
    user_id_input = layers.input(input_shape=(1,), dtype=tf.string, name='user_id_input')

    #TODO: why EMPTY? We just create input layers. It doesn't contain any data
    movie_id_input = layers.input(input_shape=(1,), dtype=tf.string, name='movie_id_input')

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

    # add other user/movie information as input? positional encoding?

    return user_id_input, sequence_movie_ids_input, target_movie_id_input, sequence_ratings_input, target_rating_input

def encode_input(inputs, sequence_length=4):
    # TODO: difference between input and dataframe... is vocabulary generated outside?
    # vocabulary = create_embedding_vocabulary(inputs) 
    ## TODO: Encode user features - what features do we include, how to separate
    users = input['user_id_input']
    vocabulary = list(users.user_id.unique())
    for user_features in users:
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{user_features}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder)

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    movies = input['movie_id_input']

    ## Create a movie embedding encoder
    unique_movie_values = list(movies.movie_id.unique()),
    movie_vocabulary = unique_movie_values
    movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    movie_index_lookup = StringLookup(
        vocabulary=movie_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="movie_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        name=f"movie_embedding",
    )
    # Create a vector lookup for movie genres.
    genre_vectors = movies_data['movie_genre'].to_numpy()
    movie_genres_lookup = layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(genre_vectors),
        trainable=False,
        name="genres_vector",
    )
    # Create a processing layer for genres.
    movie_embedding_processor = layers.Dense(
        units=movie_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )

    ## Define a function to encode a given movie id.
    def encode_movie(movie_id):
        # Convert the string input values into integer indices.
        movie_idx = movie_index_lookup(movie_id)
        movie_embedding = movie_embedding_encoder(movie_idx)
        encoded_movie = movie_embedding
        movie_genres_vector = movie_genres_lookup(movie_idx)
        encoded_movie = movie_embedding_processor(
            layers.concatenate([movie_embedding, movie_genres_vector])
        )
        return encoded_movie

    ## Encoding target_movie_id
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    ## Encoding sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=movie_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the movie.
    sequence_ratings = tf.expand_dims(inputs["sequence_ratings"], -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_movies + encodded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for encoded_movie in tf.unstack(
        encoded_sequence_movies_with_poistion_and_rating, axis=1
    ):
        encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
    encoded_transformer_features.append(encoded_target_movie)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features

## Vocaubulary dataset for Embedding
def create_embedding_vocabulary(dataframe, col_list):
    
    vocabulary = {}

    #Create vocabulary for each column
    for col in col_list:
        unique_values = dataframe[col].unique().tolist()
        vocabulary[col] = unique_values
    
    return vocabulary










    

