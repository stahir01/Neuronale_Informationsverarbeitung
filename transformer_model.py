import math
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from user_sequence import *
from input_layer_and_embedding import create_input_model, encode_input

# global variables for the architecture

hidden_units = [1024, 512, 256]
dropout_rate = 0.1
num_heads = 3

# after Paper structure 
def create_model():
    inputs = create_input_model()
    transformer_features, other_features = encode_input(inputs)

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    # Dropout prevents overfitting!
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    # feed forward
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)

    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    # Does that work?
    if other_features is not None:
        features = layers.concatenate([features, other_features])

    # Fully-connected layers.
    # Dense - fully connected
    # BatchNormalization - stable distribution of activation values throughout training
    # Dropout - overfitting
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_model()