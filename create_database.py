# Script for making database and populating it with book data
import pandas as pd 

# loading the books from csv file to dataframe
books = pd.read_csv('data/goodbooks-10k/books.csv')

# populating book_app database with sql-alchemy
from book_app import db

db.create_all()

from book_app import Book

# looping over all the books and adding them to the database
for i in range(len(books)):
    book_to_store = Book(book_id=int(books['book_id'].iloc[i]), 
                        goodreads_id=int(books['goodreads_book_id'].iloc[i]),
                        title=books['title'].iloc[i],
                        author=books['authors'].iloc[i],
                        book_image=books['image_url'].iloc[i],
                        average_rating=float(books['average_rating'].iloc[i])
                        ) 
    db.session.add(book_to_store)
db.session.commit()
