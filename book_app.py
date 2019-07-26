import sys
from flask import Flask, render_template, url_for, redirect, request
import os
import numpy as np
import pickle 
import pandas as pd
from forms import SearchForm

app = Flask(__name__)

app.config['SECRET_KEY'] = '0b248914a4417846b62d195c17626830'

# Using the trained latent vector for predictions
x = pickle.load(open("pickled_X.p", "rb"))
# let's transpose x to 
x = x.transpose()
# loading the corresponding movies
books = pd.read_csv('data/goodbooks-10k/books.csv')

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
            books = find_similar(x, books_to_search[0][0], 100)
            return render_template('sort_by_sentiment_app.html',books=books, form=form)
        # handling of multiple results
        else:
            return render_template('sort_by_sentiment_app.html', book_list=books_to_search, form=form)

    if request.method == 'POST':
        if request.form.get('Search') == 'Search similar':
            book_index = request.form.get('index')
            books = find_similar(x, int(book_index), 100)
            return render_template('sort_by_sentiment_app.html',books=books, form=form)

    return render_template('sort_by_sentiment_app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))

