from content_based_utils import *

if __name__ == '__main__':

#### DATA PREPROCESSING ####

    # Get the files and read them into Pandas DataFrames
    movies_path = './data/movies.csv'
    movies_df = pd.read_csv(movies_path)

    ratings_path = './data/ratings.csv'
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.drop('timestamp', axis=1)

    tags_path = './data/tags.csv'
    tags_df = pd.read_csv(tags_path)
    tags_df = tags_df.drop('timestamp', axis=1)


#### CONTENT BASED IMPLEMENTATION ####

    print('\n\n*** INITIALIZATION ***\n')

    # Movies class initialization
    print('Creating movies profiles...')
    movies = Movies(movies_df, tags_df, tfidf=True, lsa=True, n_components=40)

    # Users class initialization
    print('Creating users profiles...')
    users = Users(ratings_df, movies)

    # ContentBased class initialization
    print('Creating recommender...')
    recommender = ContentBased(users, movies)
    

#### RECOMMENDATIONS ####

    print('\n\n*** RECOMMENDATIONS ***\n')

    # We predict the top k recommendations for user defined by userId
    userId = int(input('userId: '))
    k = int(input('N. of recommendations: '))
    print('It may require some time. Please wait ...\n')
    recommendations = recommender.recommend(userId=userId, n_recommendations=k)
    print(recommendations)