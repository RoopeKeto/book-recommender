#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:28:41 2019

@author: roope
"""

import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np

# # # # # Collaborative Filtering using Matrix Factorization algorithm # # # # # 
# Using bookdata from goodbooks-10k datase

# preparing goodbooks-10k data

#book_tags = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/book_tags.csv')
#books = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/ratings.csv')
#tags = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/tags.csv')
#to_read = pd.read_csv('/home/roope/projects/book-recommender/data/goodbooks-10k/to_read.csv')

# removing possible duplicate rows
#ratings.drop_duplicates(inplace=True)

# checking rating range
#min_rating = ratings.rating.min() # =1
#max_rating = ratings.rating.max() # = 5


# getting only half of the data for performance reasons
#ratings = ratings.iloc[:5000,:]

# creating matrix of shape (movies, users)
#rating_df = ratings.pivot(index='book_id', columns='user_id', values='rating')

# not taking NaN's into account...
# replacing NaN's with book average rating
#for (columName, columnData) in rating_df.iteritems():
    #mean = rating_df[columName].mean(axis=0, skipna=True)
 #   rating_df[columName] = rating_df[columName].fillna(mean)
    
# # # BUILDING THE ALGORITHM # # #
 

# getting indicies (to start from 0)
user_indecies = [x-1 for x in ratings.user_id.values]
item_indecies = [x-1 for x in ratings.book_id.values]
rates = ratings.rating.values

# latent factor multiplication where: 
# -  X = latent factor for books
# - theta = latent factor for users
num_features = 40
Theta = tf.Variable(initial_value=tf.truncated_normal([53424,num_features]), name='users')
X = tf.Variable(initial_value=tf.truncated_normal([num_features, 10000]), name='items')

result = tf.matmul(Theta, X)

# flattening the result  matrix
result_flatten = tf.reshape(result, [-1])
R = tf.gather(result_flatten,user_indecies * tf.shape(result)[1] + 
              item_indecies, name="extracting_user_rating")

# creating cost function
diff_op = tf.subtract(R, rates, name='difference')
diff_op_abs = tf.abs(diff_op, name='abs_difference')
base_cost = tf.reduce_sum(diff_op_abs, name='sum_abs_error')

# adding regularization
lda = tf.constant(.001, name='lambda')
norm_sums = tf.add(tf.reduce_sum(tf.abs(Theta, name='user_abs'), name='user_norm'),
                   tf.reduce_sum(tf.abs(X, name='item_abs'), name='item_norm'))
regularizer = tf.multiply(norm_sums, lda, 'regularizer')

# final cost function
cost = tf.add(base_cost, regularizer)

# optimizer
lr = tf.constant(.001, name='learning_rate')
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96,staircase=True)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(cost, global_step=global_step)

# running the graph
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
saver = tf.train.Saver()

sess.run(init)
cost_stored = []
for i in range(100000):
    sess.run(training_step)
    # checking the cost
    #print("cost is:", sess.run(base_cost))
    cost_to_store = sess.run(base_cost)
    cost_stored.append(cost_to_store)
    if (i % 1000) == 0:
        print(cost_to_store)
    
    # saving every 10 000 steps    
    if (i % 10000) == 0:
        save_path = saver.save(sess, "/tmp/recommender_model.ckpt")
        
# saving final model
save_path = saver.save(sess, "/tmp/recommender_model_final.ckpt")

# checking one prediction
u, p, r = ratings[['user_id', 'book_id', 'rating']].values[0]
rhat = tf.gather(tf.gather(result, u-1), p-1)
print("rating for user " + str(u) + " for item " + str(p) + 
      " is " + str(r) + " and our prediction is: " + str(sess.run(rhat)))


# saving the X  - the latent vector for books - for future use in recommendation app
import pickle
X_trained = sess.run(X)
pickle.dump( X_trained, open("pickled_X.p", "wb"))
    

