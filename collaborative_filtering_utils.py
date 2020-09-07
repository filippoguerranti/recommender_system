# Libraries import
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import KNNBasic,  KNNWithMeans, KNNBaseline, SVD
from surprise.model_selection import KFold
from tqdm import tqdm


def print_info(dataframe):
    '''Function which prints some informations about the dataset'''
    columns = []
    for c in dataframe.columns:
        columns.append(c)
    n_ratings = dataframe.shape[0]
    n_movies = dataframe[['movieId']].drop_duplicates(['movieId']).shape[0]
    n_users = dataframe[['userId']].drop_duplicates(['userId']).shape[0]
    print('\n\n*** DATASET DETAILS ***\n')
    print('Columns: ', columns)
    print('N. of ratings: ', n_ratings)
    print('N. of movies: ', n_movies)
    print('N. of users: ', n_users)

def write_results(results_path, results):
    '''Function which writes the "results" of the 5-fold cross-validation
    on a .csv file having path "results_path"'''
    f = open(results_path, 'w')
    f.write('Algorithm')
    folds = int(sum(len(v) for v in results.values())/len(results))
    for fold in range(1,folds+1):
        f.write(',Fold '+str(fold))
    for name, result in results.items():
        f.write('\n'+name)
        for r in result:
            f.write(','+str(r))
    f.close()
    results = pd.read_csv(results_path)
    results['Mean'] = results.mean(axis=1)
    results['Std'] = results.std(axis=1)
    results = results.set_index('Algorithm')
    results.to_csv(results_path)

def write_predictions(predictions_path, predictions):
    '''Function which writes the "predictions" of the best algorithm
    on a .csv file having path "predictions_path"'''
    f = open(predictions_path, 'w')
    f.write('userId,movieId,est_rating')
    for userId, movieId, _, est_rating, _ in predictions:
        f.write('\n'+str(userId)+','+str(movieId)+','+str(round(est_rating,2)))
    f.close()

def get_k_top_recommendations(userId, k, predictions, movies):
    '''Function which return the k top recommendations for user given by userId'''
    user_top_k = predictions[predictions['userId']==userId].sort_values(by='est_rating', ascending=False).head(k)
    user_top_k = pd.merge(left=user_top_k, right=movies, on='movieId')
    return user_top_k
        

def get_k_top_liked(userId, k, ratings, movies):
    '''Function which return the k top recommendations for user given by userId'''
    user_top_k = ratings[ratings['userId']==userId].sort_values(by='rating', ascending=False).head(k)
    user_top_k = pd.merge(left=user_top_k, right=movies, on='movieId')
    return user_top_k