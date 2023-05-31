import pandas as pd
import numpy as np


import pandas as pd

def get_ratings_per_book_data(book_file_path, rating_file_path, users_file_path):
    # Load & clean data
    book_data = (
        pd.read_csv(book_file_path, low_memory=False)
        .drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
        .rename(columns={'ISBN': 'isbn', 'Book-Title': 'book_title', 'Book-Author': 'book_author', 'Year-Of-Publication': 'publication_year', 'Publisher': 'publisher'})
    )

    rating_data = (
        pd.read_csv(rating_file_path, low_memory=False)
        .rename(columns={'User-ID': 'user_id', 'Book-Rating': 'book_rating', 'ISBN': 'isbn'})
    )

    users_data = (
        pd.read_csv(users_file_path, low_memory=False)
        .drop(columns=['Location'])
        .rename(columns={'User-ID': 'user_id', 'Age':'age'})
    )

    # Merge data
    ratings_per_book = (
        rating_data.merge(book_data, on='isbn', how='inner')
        .groupby(['user_id', 'isbn', 'book_title', 'book_author', 'publication_year', 'publisher', 'book_rating'])
        .size()
        .reset_index(name='count')
        .sort_values(['user_id', 'book_rating'])
        .drop(columns=['count'])
    )

    return ratings_per_book, book_data, rating_data, users_data



if __name__ == '__main__':
    book_file_path = 'book-dataset/Books.csv'
    rating_file_path = 'book-dataset/Ratings.csv'
    users_file_path = 'book-dataset/Users.csv'

    ratings_per_book, book_data, rating_data, users_data  = get_ratings_per_book_data(book_file_path, rating_file_path, users_file_path)
    print(f"Rating per book: \n {ratings_per_book.head(10)}\n")
    print(f"Book Information: \n {book_data.head(10)}\n")

