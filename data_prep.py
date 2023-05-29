import pandas as pd
import pandasql as ps

import pandas as pd
import pandasql as ps

def process_book_data(books_data, rating_data, users_data):
    # Load the CSV files
    books_df = pd.read_csv(books_data)
    rating_df = pd.read_csv(rating_data)
    users_df = pd.read_csv(users_data)
    
    # Drop unnecessary columns and rename columns
    books_df = books_df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    books_df = books_df.rename(columns={'Book-Title': 'books_title', 'Book-Author': 'books_author', 'Year-Of-Publication': 'publication_year', 'Publisher': 'publisher'})
    users_df = users_df.drop(columns=['Location'])
    users_df = users_df.rename(columns={'User-ID': 'user_id', 'Age': 'age'})
    rating_df = rating_df.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'book_rating'})
    
    # Perform SQL query to join the tables and select desired columns
    query = """
        SELECT 
            rating_data.user_id AS user_id,
            rating_data.ISBN AS ISBN,
            books_data.books_title AS book_title,
            books_data.books_author AS books_author,
            books_data.publication_year AS publication_year,
            books_data.publisher AS publisher,
            rating_data.book_rating AS book_rating
        FROM rating_data
        INNER JOIN books_data ON rating_data.ISBN = books_data.ISBN
    """
    ratings_per_book = ps.sqldf(query)
    
    # Return the processed books data and ratings per book
    return books_df, ratings_per_book

# Example usage
books_data = 'book-dataset/Books.csv'
rating_data = 'book-dataset/Ratings.csv'
users_data = 'book-dataset/Users.csv'

processed_books_data, ratings_df = process_book_data(books_data, rating_data, users_data)
print("Processed Books Data:")
print(processed_books_data)
print("\nRatings per Book:")
print(ratings_df)



