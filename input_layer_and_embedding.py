import math
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split   
from data_prep import transform_data
from user_sequence import *
from data_split import tensor_batched_dataset

USER_FEATURES = []

def create_input_model(sequence_size=4):

    # Define input layer for user IDs
    user_id_input = layers.Input(
        name="user_id", shape=(1,), dtype=tf.string
    )

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

# IDEA: 
# - create a single input tensor from the user informations (features) 
# - encode movies in the sequence and add movie genre information (encoded as vectors)
# - positional embedding added to the movie sequences
# - add target movie to the movie sequence embedding
def encode_input(inputs, sequence_length=4):

    # create vocabularies for embedding
    vocabulary = create_embedding_vocabulary(inputs) 
    encoded_features = []
    ## Encode user features - what features do we include, how to separate
    # TODO: add user information, at this point only the ID is used for encoding meaning there is no logical background to it
    user_features = ['user_id_input']
    for user_feature in user_features:
        idx = layers.StringLookup(vocabulary=vocabulary[user_feature])(
            inputs[user_feature]
        )
        embedding_dims = int(math.sqrt(len(vocabulary[user_feature])))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary['user_id_input']),
            output_dim=embedding_dims,
            name=f"{user_feature}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_features.append(embedding_encoder(idx))

    # Create a single embedding vector for the user features - DAS GEHT ANDERS
    if len(encoded_features) > 1:
        encoded_features = layers.concatenate(encoded_features)
    elif len(encoded_features) == 1:
        encoded_features = encoded_features[0]
    else:
        encoded_features = None

    ## Create a movie embedding encoder
    movie_embedding_dims = int(math.sqrt(len(vocabulary["movie_id"])))
    # Create a lookup to convert string values to integer indices.
    movie_index_lookup = layers.StringLookup(
        vocabulary=vocabulary["movie_id"],
        num_oov_indices=0,
        name="movie_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = layers.Embedding(
        input_dim=len(vocabulary["movie_id"]),
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
        movie_genres_vector = movie_genres_lookup(movie_idx)
        encoded_movie = movie_embedding_processor(
            layers.concatenate([movie_embedding, movie_genres_vector])
        )
        return encoded_movie

    ## Encoding sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)

    ## Encoding target_movie_id.
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=movie_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encoded_positions = position_embedding_encoder(positions)
    # Get sequence ratings to incorporate them into the encoding of the movie.
    sequence_ratings = tf.expand_dims(inputs["sequence_ratings"], -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_position_and_rating = layers.Multiply()(
        [(encoded_sequence_movies + encoded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for encoded_movie in tf.unstack(
        encoded_sequence_movies_with_position_and_rating, axis=1
    ):
        encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
    encoded_transformer_features.append(encoded_target_movie)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_features

## Vocabulary dataset for Embedding
def create_embedding_vocabulary(dataframe):
    
    vocabulary = {}

    #Create vocabulary for each column
    for col in dataframe:
        unique_values = dataframe[col].unique().tolist()
        vocabulary[col] = unique_values
    
    return vocabulary










    

