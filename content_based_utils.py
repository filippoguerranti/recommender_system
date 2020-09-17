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

        # Movies dataframe
        self._movies_df = movies_df

        self.__create_movies_profiles(movies_df, tags_df)
        self.__create_movies_terms(tfidf=tfidf)

        # If latent semantic analysis is True, than apply it
        if lsa:     
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

        # Dictionary containing the movies as keys and the lists of words as values
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

        # If the TF-IDF is selected, then apply it
        if tfidf:       
            counter = TfidfVectorizer()

        # Otherwise go for the classic word counting
        else:           
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

        It creates the users' profiles (vectors) based on the genres and tags of the movies each users rated
        '''

        # Ratings dataframe
        self.__ratings_df = ratings_df

        # List containing all the IDs of the users
        self.__users_list = ratings_df['userId'].unique() 
        # List containing all the IDs of the movies
        self.__movies_list = ratings_df['movieId'].unique() 

        self.__create_users_movies_dict(ratings_df)
        self.__create_users_vectors(movies_instance)

    
    def __create_users_movies_dict(self, ratings_df):

        ''' Creates a dictionary containing all the users as keys and the rated movies for each user
        as values 
        
        Params:
            - ratings_df: ratings' Pandas DataFrame

        Returns:
            - self.__users_movies_dict: nested dictionary having userId as keys and for each user 
                    the ratings are keys and the movieIds are values
        '''

        # Returned dictionary  
        self.__users_movies_dict = {}   

        # Loop over all the users in the dataset                    
        for user in self.__users_list:                      
            self.__users_movies_dict[user] = {}
            # Loop over all the ratings
            for rating in np.arange(0,5.5,0.5):
                # Create a list of movieIds for all the ratings and for all the users
                self.__users_movies_dict[user][rating] = list(ratings_df[(ratings_df['userId']==user) & (ratings_df['rating']==rating)]['movieId'])
    

    def __create_users_vectors(self, mv):

        ''' Creates the users' profiles (vectors) which is the same vector as the one
        describing the movies' profiles (vectors).

        Params:
            - mv: instance of class Movies

        Returns:
            - self.__users_vectors: dictionary containing a vector for each user

        '''

        # Dictionary containing the vector model for each user
        self.__users_vectors = {}
        # Loop over all the users
        for user in self.__users_list:
            # Numerator is the sum of all the movie-vectors multiplied by the rating
            numerator = 0
            # Denominator is the sum of all the ratings
            denominator = 0
            # Loop over all the ratings
            for rating in self.__users_movies_dict[user]:
                # Loop over all the movies of a given user having given rating
                for movie in self.__users_movies_dict[user][rating]:
                    numerator += (rating) * mv.movie_vector(movie)
                    denominator += rating
            # Weighted average
            self.__users_vectors[user] = numerator/denominator


    def movies_dict(self, userId):

        ''' Returns the dictionary describing the movies for each users.

        Params:
            - userId: ID of the user of which the movies are requested
        
        Returns:
            - self.__users_movies_dict[userId]: dictionary of movies '''

        return self.__users_movies_dict[userId]


    def not_rated_movies_list(self, userId):

        '''Returns the not rated movies of userId.

        Params:
            - userId: ID of the user
        
        Returns:
            - self.__users_not_rated_movies_dict[userId]: list containing all 
                    the not rated movies by userId
        '''
        # List containing all the movies rated by userId
        list1 = list(self.__ratings_df[(self.__ratings_df['userId']==userId)]['movieId'])
        # List containing all the movies in the dataset
        list2 = self.__movies_list
        # Returns the list containing all the not rated movies by userId
        return list(set(list2)-set(list1))


    def user_vector(self, userId):

        '''Returns the vector model of the user given by userId.

        Params:
            - userId: ID of the user of which the vector is requested
        
        Returns:
            - self.__user_vectors[userId]: vector model of the users
        '''

        return self.__users_vectors[userId]



class ContentBased():

    ''' Class to handle a content-based recommendation system '''

    def __init__(self, users_instance, movies_instance):

        ''' Class constructor.

        Params:
            - user_instance: instance of class Users
            - movies_instance: instance of class Movies

        It creates the content-based recommender system on users and movies
        passed as parameters
        '''
        self.__users = users_instance
        self.__movies = movies_instance


    def recommend(self, userId, n_recommendations=10):

        ''' Recommends movies to user defined by userId.
        
        Params:
            - userId: id of the user we want to recommend movies
            - n_recommendations: number of recommendations
            
        Output:
            - recommendations: list of list in which the first element
                    of each sub-list represents the similarity measure 
                    and the second represents the movieId
        '''

        # List containing the similarity and the movieId
        recommendations = [[0,0]]*n_recommendations
        user_vector = self.__users.user_vector(userId).reshape(1,-1)
        for movieId in self.__users.not_rated_movies_list(userId):
            movie_vector = self.__movies.movie_vector(movieId).reshape(1,-1)
            # Compute the cosine similarity between the user profile and the movie profile
            sim = cosine_similarity(user_vector, movie_vector)
            # Since the recommendations are sorted in ascending order this means that
            # if sim > recommendations[0][0] than sim must be inserted in the list
            if sim > recommendations[0][0]:
                recommendations[0] = [sim, movieId]
                recommendations = sorted(recommendations, key=lambda x: x[0])
        
        # Arrange recommendations in a more readable form
        for i in range(n_recommendations):
            sim = round(recommendations[i][0][0][0],3)
            movie_id = self.__movies._movies_df[self.__movies._movies_df['movieId']==recommendations[i][1]].values[0][0]
            movie_title = self.__movies._movies_df[self.__movies._movies_df['movieId']==recommendations[i][1]].values[0][1]
            movie_genres = self.__movies._movies_df[self.__movies._movies_df['movieId']==recommendations[i][1]].values[0][2]
            recommendations[i][0] = movie_id
            recommendations[i][1] = sim
            recommendations[i].append(movie_title)
            recommendations[i].append(movie_genres)
        
        recommendations = pd.DataFrame(recommendations, columns=['movieId','similarity','title','genres'])
        
        return recommendations