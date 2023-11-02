import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def load_original_data(file_path="../artifacts/fetal_health.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_preprocessed_data(file_path="../artifacts/fetal_health_preprocessed.pkl") -> pd.DataFrame: 
    df = pd.read_pickle(file_path)
    return df

def id_outliers(df:pd.DataFrame)-> pd.DataFrame:
    """
    Identify outliers for each column of a dataframe
    :param df: dataframe
    :return: dataframe with lower and upper bound and number of outliers
    """
    # Initialize a list to store data for the new DataFrame
    result_data = []
    for col_name in df.columns:
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = len(df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)])
        result_data.append([lower_bound, upper_bound, n_outliers])
    # while list.append is amortized O(1) at each step of the loop, pandas' concat is O(n) at each step.
    # making it inefficient when repeated insertion is performed (new DataFrame is created for each step).
    # So a better way is to append the data to a list and then create the DataFrame in one go.
    outliers = pd.DataFrame(result_data, columns=['lower_bound', 'upper_bound', 'n_outliers'], index=df.columns)
    return outliers

def split_data(df:pd.DataFrame) -> tuple:
    """
    Split data into train and test set
    :param df: dataframe
    :return: train and test set
    """
    # features
    X = df.drop('fetal_health', axis=1)
    # target
    y = df['fetal_health']
    #  discretize the target variable
    y = pd.cut(y, bins=3, labels=[0,1,2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13,shuffle=True)
    return X_train, X_test, y_train, y_test

def algorithm_comparison(cv_results_accuracy:dict, pipe_dict:dict) -> None:
    """
    Compare the accuracy of different algorithms
    :param cv_results_accuracy: dictionary with the accuracy of each algorithm
    :param pipe_dict: dictionary with the name of each algorithm
    :return: None
    """
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(cv_results_accuracy)
    ax.set_xticklabels(pipe_dict.values())
    plt.show()

def plot_learning_curve(model:object, title:str, X:pd.DataFrame, y:pd.DataFrame,
                        ylim:tuple=None, cv:int=None, n_jobs:int=None,
                        train_sizes:tuple=np.linspace(.1, 1.0, 5)) -> plt:
    """
    Generate a simple plot of the test and training learning curve.
    :param model: model
    :param title: title of the plot
    :param X: features
    :param y: target
    :param ylim: tuple with the limits of the y axis
    :param cv: cross validation
    :param n_jobs: number of jobs
    :param train_sizes: train sizes
    :return: plot
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training dataset size")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def feature_importance(model:object, X_train:pd.DataFrame, y_train:pd.DataFrame,
                         df:pd.DataFrame, title:str) -> tuple:
    """
    Plot feature importance
    :param model: model
    :param X_train: features
    :param y_train: target
    :param df: dataframe
    :param title: title of the plot
    :return: best features and highest importances
    """
    # fit the model
    model.fit(X_train, y_train)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (df.columns[i],v))
    # 5 most important features
    idx = np.argsort(importance)[::-1][:5]
    highest_importances = importance[idx]
    best_features = df.columns[idx]
    # plot feature importance
    fig = plt.figure(figsize=(50, 35))
    plt.title(title, fontsize=50)
    plt.bar([x for x in df.columns[:-1]], importance)
    plt.show()
    return best_features, highest_importances
    