import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import Normalizer, StandardScaler


def load_and_preprocess_data(datafile="credit_data.zip", seed=None):
    # Set up path for datafile
    cur_dir = os.path.abspath(os.getcwd())
    datapath = os.path.join(cur_dir, datafile)

    # Load data
    data = pd.read_csv(datapath, index_col=0)
    data.dropna(inplace=True)

    # Separate features and outcome
    features = data.drop("SeriousDlqin2yrs", axis=1)
    outcomes = data["SeriousDlqin2yrs"].values

    # Standardize features
    scaler = preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)


    # Balance the dataset
    rng = np.random.default_rng(seed)
    default_indices = np.where(outcomes == 1)[0]
    other_indices = np.where(outcomes == 0)[0][:10000]
    indices = np.concatenate((default_indices, other_indices))
    features_balanced = features_scaled[indices]
    outcomes_balanced = outcomes[indices]

    # Shuffle the dataset
    shuffled_indices = rng.permutation(len(indices))
    features_shuffled = features_balanced[shuffled_indices]
    outcomes_shuffled = outcomes_balanced[shuffled_indices]

    # Combine back into a DataFrame
    data_preprocessed = pd.DataFrame(features_shuffled, columns=[*data.columns.drop("SeriousDlqin2yrs")])
    outcomes_shuffled = np.where(outcomes_shuffled == 0, -1, 1)

    data_preprocessed['Outcome'] = outcomes_shuffled

    return data_preprocessed

def load(dataset="Houses"):

    X, y = fetch_openml(name='houses', version=2, return_X_y=True, as_frame=True)
    print(X)
    #c = np.unique(y)    
    #y[y==c[0]] = -1
    #y[y==c[1]] = 1
    y = y.replace('P', 1)
    y= y.replace('N', -1)
    features = X.values


    standardizer = StandardScaler()
    pf = standardizer.fit_transform(features)
    p = pd.DataFrame(pf, columns=[*X.columns])
    p['Outcome'] = y
    print(p)
    return p



# Execute the function and save the results
processed_data = load_and_preprocess_data(seed=0)
processed_data.to_csv("processed_credit_data.csv", index=False)

alter = load()
alter.to_csv("processed_houses.csv", index=False)