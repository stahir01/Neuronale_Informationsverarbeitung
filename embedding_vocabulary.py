from data_prep import transform_data
from user_sequence import *


def create_embedding_vocabulary(dataframes = [], col_list = []):

    """
    Create a vocabulary for the specified columns in the given dataframes.

    Args:
        dataframes (list): A list of pandas DataFrames containing the data.
        col_list (list): A list of column names to create the vocabulary for.

    Returns:
        vocabulary (dict): A dictionary where the keys are column names and the values are lists of unique values in each column.

    """

    vocabulary = {}

    #Create vocabulary for each column
    for dataframe in dataframes:
        for col in col_list:
            if col in dataframe.columns:  # Check if the column exists in the current dataframe
                if col not in vocabulary: # Check if the column has already been processed
                    unique_values = dataframe[col].unique().tolist()     # Extract unique values from the column
                    vocabulary[col] = unique_values                      # Add the column and its unique values to the vocabulary
    
    return vocabulary

if __name__== '__main__':
    movies_data, ratings_data, users_data = transform_data()
    rating_sequence = prepare_user_sequence_data(ratings_data)
    vocabulary = create_embedding_vocabulary([movies_data, ratings_data, users_data], ['user_id', 'movie_id', 'age', 'occupation', 'gender']) #We need to create vocabulary for original data
    print(vocabulary)


##Explanation:
    # This code demonstrates how to create a vocabulary for an embedding layer using the initial data. The purpose of the
    # vocabulary is to map categorical variables to unique integer identifiers. The code operates on multiple dataframes,
    # and since some columns are shared among the dataframes, it avoids redundant processing by selecting those columns
    # only once.

    # The function `create_embedding_vocabulary` takes a list of dataframes and a list of column names as inputs. It
    # iterates over each dataframe and each selected column, checking if the column exists in the dataframe. If the column
    # is found and has not been processed before, it extracts the unique values from that column and adds them to the
    # vocabulary dictionary, with the column name as the key and the list of unique values as the value.

    # By specifying the column list, the code allows the user to choose which columns to include in the vocabulary. This
    # is useful when dealing with dataframes that contain both categorical and numerical columns. Numerical columns, such
    # as the 'rating' column, do not need to be embedded since they are already represented as numeric values.

    # Finally, the resulting vocabulary is printed, providing a mapping of column names to lists of unique values found
    # in each column. This vocabulary can then be used in the embedding layer for further data processing or modeling.





