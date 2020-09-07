from collaborative_filtering_utils import *

if __name__ == '__main__':

#### DATA PREPROCESSING ####

    # Get the file and read it into a Pandas DataFrame
    movies_path = './data/movies.csv'
    movies_df = pd.read_csv(movies_path)
    ratings_path = './data/ratings.csv'
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.drop('timestamp', axis=1)

    # Print some informations about the dataset
    print_info(dataframe=ratings_df)


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
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    results = {}
    for algo_name, algo in tqdm(algos.items()):
        rmse_list = []
        for trainset, testset in kf.split(data):
            # Train and test algorithm
            predictions = algo.fit(trainset).test(testset)
            # Compute and store Root Mean Squared Error
            rmse = accuracy.rmse(predictions, verbose=False)
            rmse_list.append(rmse)
        results[algo_name] = rmse_list
    

#### RESULTS ####

    results_path = './results/results.csv'
    write_results(results_path=results_path,results=results)
    results_df = pd.read_csv(results_path)

    print('\n\n*** RESULTS ***\n',results_df)


#### PREDICTIONS ####

    print('\n\n*** PREDICTIONS ***\n')

    # We take the algorithm that performed best and use it to predict all the missing pairs
    best_algo = results_df[results_df['Mean']==results_df['Mean'].min(axis=0)]['Algorithm'].to_numpy()[0]
    pred_algo = algos[best_algo]

    # We train the algorithm with the whole training set
    trainset = data.build_full_trainset()
    pred_algo.fit(trainset)

    # We test the algorithm over the remaining set (to find the missing ratings)
    anti_testset = trainset.build_anti_testset()
    predictions = pred_algo.test(anti_testset)

    # We write the predictions on a .csv file
    predictions_path = './results/predictions.csv'
    write_predictions(predictions_path=predictions_path, predictions=predictions)

    # We get the top k predictions for user x
    userId = int(input('userId: '))
    k = int(input('N. of recommendations: '))
    predictions_df = pd.read_csv(predictions_path)
    user_pred = get_k_top_recommendations(userId=userId, k=k, predictions=predictions_df, movies=movies_df)
    print(f'\n\nTop {k} recommendations for user {userId}\n')
    print(user_pred)

    # We print the k top liked movies of the same user
    user_liked = get_k_top_liked(userId=userId, k=k, ratings=ratings_df, movies=movies_df)
    print(f'\n\nTop {k} liked movies of user {userId}\n')
    print(user_liked)