import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD



class MoviesTerms():

    def __init__(self, movies_df, tags_df, dictionary_reduction=True, n_components=40):
        self.create_movies_profiles(movies_df, tags_df)
        self.create_movies_terms_dataframe()
        if dictionary_reduction:
            self.dictionary_reduction(n_components)



    def create_movies_profiles(self, movies_df, tags_df):

        '''Creates a bag of words for each movie (a.k.a. the movie-profile).
        An item (movie) profile is a set of features: in this case genres and tags.

        Params:
            - movies_df: movies Pandas DataFrame
            - tags_df: tags Pandas DataFrame

        Returns:
            - movies_profiles: dictionary containing a string with all the words for each movie
                    { movie1: ' children drama in netflix queue',
                    movie2: ' action adventure fantasy mystery',
                    ... }
        '''

        self.movies_profiles = {}

        # Get the words from the movies genres
        for _, row in movies_df.iterrows():
            movie_id = int(row['movieId'])
            genres = row['genres'].split('|')
            if movie_id not in self.movies_profiles:
                self.movies_profiles[movie_id] = ''
            for genre in genres:
                self.movies_profiles[movie_id] += ' ' + genre.lower()

        # Get the words from the tags of the movies
        for _, row in tags_df.iterrows():
            movie_id = int(row['movieId'])
            tag = row['tag']
            if type(tag) == str:
                self.movies_profiles[movie_id] += ' ' + tag.lower()



    def create_movies_terms_dataframe(self):
        
        '''Creates the movies-terms matrix.
        In the case of the MovieLens dataset, this consists of a (n x m) Pandas DataFrame.
        Here, n is the number of movies and m the number of total words describing them.

        Params:
            - movies_profiles: dictionary containing a string with all the words for each movie
                    { movie1: ' children drama in netflix queue',
                    movie2: ' action adventure fantasy mystery',
                    ... }
        
        Returns:
            - A: document collection to which the TF-IDF term weighting is applied
                Pandas DataFrame where rows are movies and columns are term

        '''

        counter = TfidfVectorizer()
        matrix = counter.fit_transform(self.movies_profiles.values())
        self.movies_terms_dataframe = pd.DataFrame(matrix.todense(), index=list(self.movies_profiles.keys()), columns=counter.get_feature_names())



    def dictionary_reduction(self, n_components=40):
        '''Compute the dictionary reduction on the input DataFrame A.
        It applies the SVD truncated at n_components and returns a reduced DataFrame.

        Params:
            - A: document collection to which the TF-IDF term weighting is applied
                Pandas DataFrame where rows are movies and columns are term
            - n_components: number of components of the reduced matrix
        
        Returns:
            - As: reduced DataFrame having n_components as a dimension
        '''

        movies_id = self.movies_terms_dataframe.index
        self.movies_terms_matrix = self.movies_terms_dataframe.to_numpy()
        svd = TruncatedSVD(n_components=n_components)
        self.reduced_movies_terms_matrix = svd.fit_transform(self.movies_terms_matrix)
        self.reduced_movies_terms_dataframe = pd.DataFrame(self.reduced_movies_terms_matrix, index=movies_id)