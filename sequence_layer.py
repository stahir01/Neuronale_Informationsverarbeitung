import numpy as np
import pandas as pd
from data_preperation import get_ratings_per_book_data


ratings_per_book, book_data, rating_data, users_data  = get_ratings_per_book_data('book-dataset/Books.csv', 'book-dataset/Ratings.csv', 'book-dataset/Users.csv')

# Transform movie rating into sequence
ratings_per_book = ratings_per_book.head(1000)

ratings_group = ratings_per_book.sort_values(by=["book_rating"]).groupby("user_id")
ratings_data = pd.DataFrame(    
    data={
          "user_id": list(ratings_group.groups.keys()),        
          "book_ids": list(ratings_group.isbn.apply(list)),        
          "ratings": list(ratings_group.book_rating.apply(list)),  
          })

print(ratings_data)