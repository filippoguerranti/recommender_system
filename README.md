# Recommender System
Implementation of a recommender system based on the MovieLens dataset.

---

## Outline:
* [Dataset](#dataset)
* [Collaborative Filtering](#collaborative-filtering)
* Content based

---

## Dataset
I have used the [MovieLens dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) and in particular I focused on the `movies.csv`, the `tags.csv` and the `ratings.csv` files.
The dataset contains 100863 ratings and 3683 tags provided by 610 users about 9742 movies.

---

## Collaborative Filtering
The collaborative filtering approach is based on the idea that movies which have been rated high from a cluster of users may be rated high from users which are "similar" to the ones in the cluster.
In this project I exploit the Surprise library. 

So, I:
* preprocessed the dataset using Pandas DataFrame
* created the Surprise model using different algorithms
* trained and tested all the algorithms using the 5-fold cross-validation technique
* measured the accuracy of each algorithm via RMSE
* stored the results
* made predictions