import pandas as pd
from data_prep import transform_data



def user_sequence(ratings_data):
    
    user_sequence = {}

    for index, row in ratings_data.iterrows():
        user_id = row['user_id']
        movie_ids = row['movie_id']
        ratings = row['movie_rating']
        time_period = row['time_period']

        
        if user_id not in user_sequence:
            user_sequence[user_id] = {'movies_id': [], 'movie_ratings': [], 'time_period': []}
        
        user_sequence[user_id]['movies_id'].append(movie_ids)
        user_sequence[user_id]['movie_ratings'].append(ratings)
        user_sequence[user_id]['time_period'].append(time_period)

    #Convert into dataframe
    user_sequence = pd.DataFrame(user_sequence).T
    user_sequence.index.name = 'user_id'
    user_sequence.reset_index(inplace=True)

    return user_sequence



def create_sequences(values, window_size, step_size):
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



if __name__ =='__main__':
    movies_data, ratings_data, users_data = transform_data()
    user_sequence = user_sequence(ratings_data)

    sequence_length, step_size = 4, 2


    #We already had data in asceding order according to their time_stamp so we don't need time_stamp anymore
    #However I will still keep it just to see what happens if I add time_stamp when performing positional embedding
    user_sequence.movies_id = user_sequence.movies_id.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
    )   

    user_sequence.movie_ratings = user_sequence.movie_ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
    )  

    user_sequence.time_period = user_sequence.time_period.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
    )    




