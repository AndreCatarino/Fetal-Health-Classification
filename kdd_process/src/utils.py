import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
