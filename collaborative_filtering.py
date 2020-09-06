from CollaborativeUtils import *

if __name__ == '__main__':

#### DATA PREPROCESSING ####

    # Get the file and read it into a Pandas DataFrame
    ratings_path = './data/ratings.csv'
    ratings_df = pd.read_csv(ratings_path)

    # Print some informations about the dataset
    columns = []
    for c in ratings_df.columns:
        columns.append(c)
    n_ratings = ratings_df.shape[0]
    n_movies = ratings_df[['movieId']].drop_duplicates(['movieId']).shape[0]
    n_users = ratings_df[['userId']].drop_duplicates(['userId']).shape[0]
    print('\n\n*** DATASET DETAILS ***\n')
    print('Columns: ', columns)
    print('N. of ratings: ', n_ratings)
    print('N. of movies: ', n_movies)
    print('N. of users: ', n_users)


#### COLLABORATIVE FILTERING IMPLEMENTATION ####

    # Surprise model creation: we have to define a Reader and the Dataset modules
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    # Algorithms definitions
    algos = {'KNN': KNNBasic(verbose=False), 
             'KNN-cosine': KNNWithMeans(sim_options={'name':'cosine'}, verbose=False), 
             'KNN-pearson': KNNWithMeans(sim_options={'name':'pearson'}, verbose=False),
             'KNN-cosine-baseline': KNNBaseline(sim_options={'name':'cosine'}, verbose=False),
             'KNN-pearson-baseline': KNNBaseline(sim_options={'name':'pearson'}, verbose=False),
             'SVD': SVD()}

    # 5-fold Cross-validation on each algorithm
    print('\n\n*** TRAINING AND TESTING ***\n')
    print('We are going to train and test different algorithms using the 5-fold cross-validation approach.')
    print('We will test:\n  * KNN\n  * KNN-cosine\n  * KNN-pearson\n  * KNN-cosine-baseline\n  * KNN-pearson-baseline\n  * SVD\n')
    kf = KFold(n_splits=5)
    results = {}
    for algo_name, algo in tqdm(algos.items()):
        rmse_list = []
        for trainset, testset in kf.split(data):
            # Train and test algorithm
            predictions = algo.fit(trainset).test(testset)
            # Compute and print Root Mean Squared Error
            rmse = accuracy.rmse(predictions, verbose=False)
            rmse_list.append(rmse)
        results[algo_name] = rmse_list
    
    results_path = 'data/results.csv'

    f = open(results_path, 'w')
    f.write('Algorithm')
    for fold in range(1,len(results)):
        f.write(',Fold '+str(fold))
    for name, result in results.items():
        f.write('\n'+name)
        for r in result:
            f.write(','+str(r))
    f.close()

    results = pd.read_csv(results_path)
    results['Mean'] = results.mean(axis=1)
    results['Std'] = results.std(axis=1)

    print('\n\n',results)