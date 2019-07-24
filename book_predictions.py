#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:06:25 2019

@author: roope
"""

# # # # # Script for predicting similar movies # # # # #
import pickle
import numpy as np
from operator import itemgetter

# Using the trained latent vector for predictions
x = pickle.load( open("pickled_X.p", "rb"))
# let's transpose x to 
x = x.transpose()
# loading the corresponding movies
books = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/books.csv')

# the books by indices and vector items correspond now to each other exactly. 
# x[0] is latent vector for 

# now let's choose book 2 (index 1)

chosen_book = books.iloc[1]



# let's find 10 most similar books

# 1. first calculating euclidean distances (or more specifically frobenius norm)
# applying the frobenius distance function
def find_similar(book_vectors, chosen_book_index, num_of_similar):
    chosen_book_vector = book_vectors[chosen_book_index]
    distances = []
    for i in range(len(book_vectors)):
        dist = np.linalg.norm(book_vectors[i] - chosen_book_vector)
        distances.append(dist)
    
    # sort distances from smallest to largest and take the indexes
    sorted_indexes = sorted(range(len(distances)), key=lambda k: distances[k])
    
    # 10 most similar
    most_similar = sorted_indexes[:num_of_similar]
    
    # print the names of those books:
    for i in range(len(most_similar)):
        print(books['title'].iloc[most_similar[i]])

# creating search functionality
def find_books(search_word):
    results = []
    for i in range(len(x)):
        word_to_search = books['title'].iloc[i]
        if(search_word.lower() in word_to_search.lower()):
            results.append([f'index: {i}', word_to_search])

    for i in range(len(results)):
        print(results[i][0],",",results[i][1])