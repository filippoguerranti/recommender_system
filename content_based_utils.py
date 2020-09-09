import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD



class Movies():

    ''' Class to handle Movies for the content-based recommender system.'''

    def __init__(self, movies_df, tags_df, tfidf=True, lsa=True, n_components=40):

        ''' Class constructor.

        Params:
            - movies_df: movies' Pandas DataFrame
            - tags_df: tags' Pandas DataFrame
            - tfidf: (bool) if True the TF-IDF is applied
            - lsa: (bool) if True the dictionary reduction is applied
            - n_components: number of components of the reduced space (if lsa=True)

        It creates the movies' profiles based on the genres and tags, it applies the 
        term-weighting and the dimensionality reduction (if they are True).
        '''

        self.__create_movies_profiles(movies_df, tags_df)
        self.__create_movies_terms(tfidf=tfidf)
        if lsa:     # if latent semantic analysis is True, than apply it
            self.__latent_semantic_analysis(n_components)


    def __create_movies_profiles(self, movies_df, tags_df):

        '''Creates a bag of words for each movie (a.k.a. the movie-profile).
        An item (movie) profile is a set of features: in this case genres and tags.

        Params:
            - movies_df: movies Pandas DataFrame
            - tags_df: tags Pandas DataFrame

        Returns:
            - self._movies_profiles: dictionary containing a string with all the words for each movie
                    { movie1: ' children drama in netflix queue',
                      movie2: ' action adventure fantasy mystery',
                      ... }
        '''

        # dictionary containing the movies as keys and the lists of words as values
        self.__movies_profiles = {}     

        # Get the words from the movies genres
        for _, row in movies_df.iterrows():
            movie_id = int(row['movieId'])
            genres = row['genres'].split('|')
            if movie_id not in self.__movies_profiles:
                self.__movies_profiles[movie_id] = ''
            for genre in genres:
                self.__movies_profiles[movie_id] += ' ' + genre.lower()

        # Get the words from the tags of the movies
        for _, row in tags_df.iterrows():
            movie_id = int(row['movieId'])
            tag = row['tag']
            if type(tag) == str:
                self.__movies_profiles[movie_id] += ' ' + tag.lower()


    def __create_movies_terms(self, tfidf=True):
        
        '''Creates the movies-terms matrix.
        In the case of the MovieLens dataset, this consists of a (n x m) Pandas DataFrame.
        Here, n is the number of movies and m the number of total words describing them.

        Params:
            - self.__movies_profiles: dictionary containing a string with all the words for each movie
                    { movie1: ' children drama in netflix queue',
                      movie2: ' action adventure fantasy mystery',
                      ... }
            - tfidf: (bool) if True the TF-IDF is applied
        
        Returns:
            - self.__movies_terms_dataframe: Pandas DataFrame where rows are movies and columns are term

        '''

        if tfidf:       # if the TF-IDF is selected, then apply it
            counter = TfidfVectorizer()
        else:           # otherwise go for the classic word counting
            counter = CountVectorizer()   
        matrix = counter.fit_transform(self.__movies_profiles.values())
        # Convert the matrix into a Pandas DataFrame
        self.__movies_terms_dataframe = pd.DataFrame(matrix.todense(), index=list(self.__movies_profiles.keys()), columns=counter.get_feature_names())


    def __latent_semantic_analysis(self, n_components=40):

        '''Computes the dictionary reduction on the input DataFrame A.
        It applies the SVD truncated at n_components and returns a reduced DataFrame.

        Params:
            - self.__movies_terms_dataframe: Pandas DataFrame where rows are movies and columns are term
            - n_components: number of components of the reduced matrix
        
        Returns:
            - self.__reduced_terms_dataframe: reduced DataFrame having n_components as a dimension
        '''

        # Retrive the IDs of the movies from the movies-terms dataframe
        movies_id = self.__movies_terms_dataframe.index
        # Convert the DataFrame into a numpy matrix
        self.__movies_terms_matrix = self.__movies_terms_dataframe.to_numpy()
        # Apply the Truncated Singular Value Decomposition
        svd = TruncatedSVD(n_components=n_components)
        self.__reduced_movies_terms_matrix = svd.fit_transform(self.__movies_terms_matrix)
        # Convert the matrix into a Pandas DataFrame
        self.__reduced_movies_terms_dataframe = pd.DataFrame(self.__reduced_movies_terms_matrix, index=movies_id)


    def profiles(self):

        '''Returns the movies' profiles'''

        return self.__movies_profiles

    
    def movies_terms_df(self):

        ''' Returns the movies-terms dataframe'''

        return self.__movies_terms_dataframe

    
    def movie_vector(self, movieId):

        '''Returns the vector model of the movie given by movieId.

        Params:
            - movieId: ID of the movie of which the vector is required
        
        Returns:
            - vector: vector model of the movie
        '''

        return self.__movies_terms_dataframe.loc[movieId].to_numpy()

    
    def terms_size(self):

        '''Return the dimensionality of the vector space (n. of terms)'''

        return self.__movies_terms_dataframe.shape[1]



class Users():

    ''' Class to handle Users for the content-based recommender system.'''

    def __init__(self, ratings_df, movies_instance):

        ''' Class constructor.

        Params:
            - ratings_df: ratings' Pandas DataFrame
            - movies_instance: instance of class Movies

        It creates the users' profiles based on the genres and tags of the movies each users rated
        '''

        self.__create_users_movies_dict(ratings_df)
        self.__create_users_profile(movies_instance)

    
    def __create_users_movies_dict(self, ratings_df):

        ''' Creates a dictionary containing all the users as keys and the rated movies for each user
        as values 
        
        Params:
            - ratings_df: ratings' Pandas DataFrame
        '''

        # List containing all the IDs of the users
        self.__users_list = ratings_df['userId'].unique() 
        # Returned dictionary  
        self.__users_movies_dict = {}   
        # Loop over all the users in the dataset                    
        for user in self.__users_list:                      
            self.__users_movies_dict[user] = {}
            for rating in np.arange(0,5.5,0.5):
                self.__users_movies_dict[user][rating] = list(ratings_df[(ratings_df['userId']==user) & (ratings_df['rating']==rating)]['movieId'])
    

    def __create_users_profile(self, mv):

        self.__users_profiles = {}
        for user in self.__users_list:
            numerator = 0
            denominator = 0
            for rating in self.__users_movies_dict[user]:
                for movie in self.__users_movies_dict[user][rating]:
                    numerator += (rating) * mv.movie_vector(movie)
                    denominator += rating
            self.__users_profiles[user] = numerator/denominator


    def users_movies_dict(self):
        return self.__users_movies_dict


    def profiles(self):
        return self.__users_profiles
        