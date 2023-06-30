import pandas as pd
from data_prep import transform_data


# Generate sequence of various users
def prepare_user_sequence_data(ratings_data):

    user_sequence = {}

    for index, row in ratings_data.iterrows():
        user_id = row['user_id']
        movie_ids = row['movie_id']
        ratings = row['rating']
        time_period = row['unix_timestamp']


        if user_id not in user_sequence:
            user_sequence[user_id] = {'unix_timestamp': [], 'movie_ids': [], 'ratings': []}

        user_sequence[user_id]['movie_ids'].append(movie_ids)
        user_sequence[user_id]['ratings'].append(ratings)
        user_sequence[user_id]['unix_timestamp'].append(time_period)

    #Convert into dataframe
    user_sequence = pd.DataFrame(user_sequence).T
    user_sequence.index.name = 'user_id'
    user_sequence.reset_index(inplace=True)

    return user_sequence

def prepare_user_sequence_data_second_approach(ratings_data):

    user_sequence = {}

    for index, row in ratings_data.iterrows():
        user_id = row['user_id']
        time_period = row['unix_timestamp']

        if user_id not in user_sequence:
            user_sequence[user_id] = {'unix_timestamp': [], 'movie_and_ratings': []}

        user_sequence[user_id]['movie_and_ratings'].append(row['movie_id'])
        user_sequence[user_id]['movie_and_ratings'].append(row['rating'])
        user_sequence[user_id]['unix_timestamp'].append(time_period)
        user_sequence[user_id]['unix_timestamp'].append(time_period)

    #Convert into dataframe
    user_sequence = pd.DataFrame(user_sequence).T
    user_sequence.index.name = 'user_id'
    user_sequence.reset_index(inplace=True)

    return user_sequence




# Generate subsequences from a given list of values using a sliding window approach
def generate_subsequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


# Divide subsequences for each user into separate rows in the dataframe
def append_subsequences_to_columns(dataframe, columns, sequence_length, step_size):
    for column in columns:
        for index, row in dataframe.iterrows():
            column_values = row[column]
            subsequences = generate_subsequences(column_values, sequence_length, step_size)
            dataframe.at[index, column] = subsequences

    return dataframe

# Transform the dataframe into a format that can be used for training
def transform_dataframe(dataframe, columns):
    # Explode the columns
    dataframe = dataframe.explode(columns)
    dataframe.reset_index(drop=True, inplace=True)

    # Remove the square brackets from the columns
    for col in columns:
        dataframe[col] = dataframe[col].apply(lambda x: ",".join(str(i) for i in x))

    return dataframe


if __name__ =='__main__':
    movies_data, ratings_data, users_data, all_genres = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)

    # other approach
    mixed_sequence = prepare_user_sequence_data_second_approach(ratings_data)

    sequence_length = 8
    step_size = 4
    #rating_col_subsequence = append_subsequences_to_columns(rating_sequence, ['unix_timestamp', 'movie_ids', 'ratings'], sequence_length, step_size)

    # other approach
    rating_col_subsequence = append_subsequences_to_columns(mixed_sequence, ['unix_timestamp', 'movie_and_ratings'], sequence_length, step_size)
    rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_and_ratings'])

    # rating_col_subsequence = transform_dataframe(rating_col_subsequence, ['unix_timestamp', 'movie_ids', 'ratings'])
    rating_col_subsequence = rating_col_subsequence.join(users_data.set_index("user_id"), on="user_id") #Combine with user data 
    rating_col_subsequence = rating_col_subsequence.drop(columns=['unix_timestamp', 'zip_code'])

    # rating_col_subsequence.rename(
    # columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    # inplace=True,
    # )

    print(rating_col_subsequence)