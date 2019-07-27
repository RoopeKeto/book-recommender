import sys
from flask import Flask, render_template, url_for, redirect, request, flash
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
import pickle 
import pandas as pd
from forms import SearchForm

app = Flask(__name__)

app.config['SECRET_KEY'] = '0b248914a4417wer2322d195c17626830'
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///site.db'

db =SQLAlchemy(app)

class Book(db.Model):
    book_id = db.Column(db.Integer, primary_key=True)
    goodreads_id = db.Column(db.Integer, unique=True, nullable=False)
    title = db.Column(db.String(250), nullable=False)
    author = db.Column(db.String(250), nullable=False)
    book_image = db.Column(db.String(100), nullable=False)
    average_rating = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f"Book('{self.book_id}','{self.goodreads_id}','{self.title}', '{self.author}', '{self.book_image}', '{self.average_rating}')"


# Using the trained latent vector for predictions
x = pickle.load(open("pickled_X.p", "rb"))
# let's transpose x to 
x = x.transpose()

# for fast performance, creating panda dataframe in memory from database
data_frame = pd.DataFrame(columns=['book_id', 'title', 'image_url', 'authors', 'goodreads_book_id', 'average_rating'])
books_list = []
for i in range(10000):
    bk = db.session.query(Book).get(i+1)
    book_as_list = [bk.book_id, bk.title, bk.book_image, bk.author, bk.goodreads_id, bk.average_rating]
    books_list.append(book_as_list)
books = pd.DataFrame(books_list, columns=['book_id', 'title', 'image_url', 'authors', 'goodreads_book_id', 'average_rating'])


# utility functions
def find_similar(book_vectors, chosen_book_index, num_of_similar):
    chosen_book_vector = book_vectors[chosen_book_index]
    distances = []
    for i in range(len(book_vectors)):
        dist = np.linalg.norm(book_vectors[i] - chosen_book_vector)
        distances.append(dist)
    
    # sort distances from smallest to largest and take the indexes
    sorted_indexes = sorted(range(len(distances)), key=lambda k: distances[k])
    
    # 10 most similar
    most_similar = sorted_indexes[1:num_of_similar]
    
    # print the names of those books:
    similar_books = []
    for i in range(len(most_similar)):
        book_dict = {}
        book_dict['title'] = books['title'].iloc[most_similar[i]]
        book_dict['image'] = books['image_url'].iloc[most_similar[i]]
        book_dict['author'] = books['authors'].iloc[most_similar[i]]
        book_dict['good_reads_id'] = books['goodreads_book_id'].iloc[most_similar[i]]
        book_dict['average_rating'] = books['average_rating'].iloc[most_similar[i]]
        similar_books.append(book_dict)
    return similar_books


def find_books(search_word):
    results = []
    for i in range(len(x)):
        word_to_search = books['title'].iloc[i]
        if(search_word.lower() in word_to_search.lower()):
            results.append([i, word_to_search])
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    if form.validate_on_submit():
        search_word = form.searchword.data
        books_to_search = find_books(search_word)

        # handling of no search results
        if not books_to_search:
            return render_template('sort_by_sentiment_app.html', no_results="no results", form=form)

        # handling of one match
        if len(books_to_search) == 1:
            books = find_similar(x, int(books_to_search[0][0]), 100)
            return render_template('sort_by_sentiment_app.html',books=books, form=form)
        # handling of multiple results
        else:
            return render_template('sort_by_sentiment_app.html', book_list=books_to_search, form=form)

    # handling the case of multiple results and selectin from the listing what to search
    if request.method == 'POST':
        if request.form.get('Search') == 'Search similar':
            book_index = request.form.get('index')
            books = find_similar(x, int(book_index), 100)
            return render_template('sort_by_sentiment_app.html',books=books, form=form)

    return render_template('sort_by_sentiment_app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))

