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

def create_model(
        user_features, 
        vocabulary, 
        movies_dataset, 
        genres,
        sequence_length=8, 
        hidden_units=[256, 128], 
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