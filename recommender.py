import pandas as pd
import csv
from requests import get
import json
from datetime import datetime, timedelta, date
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr

import csv
import re
import pandas as pd
import argparse
import collections
import json
import glob
import math
import os
import requests
import string
import sys
import time
import xml
import random

class Recommender(object):
    def __init__(self, training_set, test_set):
        if isinstance(training_set, str):
            # the training set is a file name
            self.training_set = pd.read_csv(training_set)
        else:
            # the training set is a DataFrame
            self.training_set = training_set.copy()

        if isinstance(test_set, str):
            # the test set is a file name
            self.test_set = pd.read_csv(test_set)
        else:
            # the test set is a DataFrame
            self.test_set = test_set.copy()

    def train_user_euclidean(self, data_set, userId):
        setEuclidean = {}
        for user in data_set[data_set.columns.difference(['movieId', userId])]:
            sub = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            dist = euclidean(sub[userId], sub[user])
            setEuclidean[user] = 1.0 / (1.0 + dist)
        return setEuclidean # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user_manhattan(self, data_set, userId):
        setManhatten = {}
        for user in data_set[data_set.columns.difference(['movieId', userId])]:
            sub = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            dist = cityblock(sub[userId], sub[user])
            setManhatten[user] = 1.0 / (1.0 + dist)
        return setManhatten # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user_cosine(self, data_set, userId):
        setCosine = {}
        for user in data_set[data_set.columns.difference(['movieId', userId])]:
            sub = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            setCosine[user] = cosine(sub[userId], sub[user])
        return setCosine # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user_pearson(self, data_set, userId):
        setPearson = {}
        for user in data_set[data_set.columns.difference(['movieId', userId])]:
            sub = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            setPearson[user] = pearsonr(sub[userId], sub[user])[0]
        return setPearson # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user(self, data_set, distance_function, userId):
        if distance_function == 'euclidean':
            return self.train_user_euclidean(data_set, userId)
        elif distance_function == 'manhattan':
            return self.train_user_manhattan(data_set, userId)
        elif distance_function == 'cosine':
            return self.train_user_cosine(data_set, userId)
        elif distance_function == 'pearson':
            return self.train_user_pearson(data_set, userId)
        else:
            return None

    def get_user_existing_ratings(self, data_set, userId):
        r = []
        for user, row in data_set.iterrows():
            if (not np.isnan(row[userId])):
                    r.append((row['movieId'], row[userId] ))
        return r 


    def predict_user_existing_ratings_top_k(self, data_set, weight, userId, k):
        weight = {k: v for k, v in sorted(weight.items(), key=lambda item: item[1], reverse=True)}
        movies = data_set[['movieId', userId]].dropna()['movieId']

        data_set = data_set.set_index('movieId')
        predicted_ratings = []

        for movie in movies:
            total_rating = 0
            total_weights = 0
            for user in list(weight.keys())[:k]:
                if not np.isnan(weight[user]) and not np.isnan(data_set.at[movie, user]):
                    total_rating += (weight[user] * data_set.at[movie, user])
                    total_weights += weight[user]


            if total_weights != 0:
                predicted_rating = total_rating / total_weights
                predicted_ratings.append((movie, predicted_rating))
        return predicted_ratings # list of tuples with movieId and rating. e.g. [(32, 4.0), (50, 4.0)]

    def evaluate(self, existing_ratings, predicted_ratings):
        # Filter out null values from both lists
        existing_ratings = {tuple[0]: tuple[1] for tuple in existing_ratings if tuple[1] is not None}
        predicted_ratings = {tuple[0]: tuple[1] for tuple in predicted_ratings if tuple[1] is not None}

        total = 0
        num = 0
        for movie in predicted_ratings:
            existing_rating = existing_ratings[movie]
            predicted_rating = predicted_ratings[movie]
            if not np.isnan(existing_rating) and not np.isnan(predicted_rating):
                total += math.pow((existing_rating - predicted_rating), 2)
                num += 1

        return {'rmse':math.sqrt(total  / num), 'ratio': (len(predicted_ratings) / len(existing_ratings))} # dictionary with an rmse value and a ratio. e.g. {'rmse':1.2, 'ratio':0.5}

    def single_calculation(self, distance_function, userId, k_values):
        user_existing_ratings = self.get_user_existing_ratings(self.test_set, userId)
        print("User has {} existing and {} missing movie ratings".format(len(user_existing_ratings), len(self.test_set) - len(user_existing_ratings)), file=sys.stderr)

        print('Building weights')
        weight = self.train_user(self.training_set[self.test_set.columns.values.tolist()], distance_function, userId)

        result = []
        for k in k_values:
            print('Calculating top-k user prediction with k={}'.format(k))
            top_k_existing_ratings_prediction = self.predict_user_existing_ratings_top_k(self.test_set, weight, userId, k)
            result.append((k, self.evaluate(user_existing_ratings, top_k_existing_ratings_prediction)))
        return result # list of tuples, each of which has the k value and the result of the evaluation. e.g. [(1, {'rmse':1.2, 'ratio':0.5}), (2, {'rmse':1.0, 'ratio':0.9})]

    def aggregate_calculation(self, distance_functions, userId, k_values):
        print()
        result_per_k = {}
        for func in distance_functions:
            print("Calculating for {} distance metric".format(func))
            for calc in self.single_calculation(func, userId, k_values):
                if calc[0] not in result_per_k:
                    result_per_k[calc[0]] = {}
                result_per_k[calc[0]]['{}_rmse'.format(func)] = calc[1]['rmse']
                result_per_k[calc[0]]['{}_ratio'.format(func)] = calc[1]['ratio']
            print()
        result = []
        for k in k_values:
            row = {'k':k}
            row.update(result_per_k[k])
            result.append(row)
        columns = ['k']
        for func in distance_functions:
            columns.append('{}_rmse'.format(func))
            columns.append('{}_ratio'.format(func))
        result = pd.DataFrame(result, columns=columns)
        return result

if __name__ == "__main__":
    recommender = Recommender("data/train.csv", "data/small_test.csv")
    print("Training set has {} users and {} movies".format(len(recommender.training_set.columns[1:]), len(recommender.training_set)))
    print("Testing set has {} users and {} movies".format(len(recommender.test_set.columns[1:]), len(recommender.test_set)))

    result = recommender.aggregate_calculation(['euclidean', 'cosine', 'pearson', 'manhattan'], "0331949b45", [1, 2, 3, 4])
    print(result)
