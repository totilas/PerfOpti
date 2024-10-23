import numpy as np
import pandas as pd
from numpy import linalg as LA


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np


def perf_houses(theta=np.zeros(8), n=10000, eps=0, seed=15, verbose=False):
    # Load data
    data = pd.read_csv('processed_houses.csv')

    # Sample n observations from the dataset
    sampled_data = data.sample(n=n, random_state=seed)
    
    if verbose:
        # Print mean of each feature for each class before modification
        means_class_minus1_before = sampled_data[sampled_data['Outcome'] == -1].mean()
        means_class_1_before = sampled_data[sampled_data['Outcome'] == 1].mean()
        
        print("Mean of features for class -1 before modification:\n", means_class_minus1_before)
        print("Mean of features for class 1 before modification:\n", means_class_1_before)
        

    u = np.zeros(8)
    u[[0,4,6]] = eps

    # Modification of the features where the outcome is -1
    mask = sampled_data['Outcome'] == -1
    sampled_data.loc[mask, sampled_data.columns[:-1]] -= u * theta

    # Split the data into features (X) and outcome (y)
    X = sampled_data.iloc[:, :-1]  # All columns except the last one (Outcome)
    y = sampled_data['Outcome']  # Only the Outcome column

    return X.values, y.values

"""
0 RevolvingUtilizationOfUnsecuredLines   -0.021904
1 age                                    -0.001603
2 NumberOfTime30-59DaysPastDueNotWorse   -0.080523
3 DebtRatio                               0.113695
4 MonthlyIncome                           0.031160
5 NumberOfOpenCreditLinesAndLoans         0.027359
6 NumberOfTimes90DaysLate                -0.037109
7 NumberRealEstateLoansOrLines            0.213597
8 NumberOfTime60-89DaysPastDueNotWorse   -0.049643
9 NumberOfDependents                     -0.030622
Outcome                                -1.000000
dtype: float64
Mean of features for class 1 before modification:
 RevolvingUtilizationOfUnsecuredLines   -0.020520
age                                    -0.411724
NumberOfTime30-59DaysPastDueNotWorse    0.090943
DebtRatio                              -0.061596
MonthlyIncome                          -0.090671
NumberOfOpenCreditLinesAndLoans        -0.141794
NumberOfTimes90DaysLate                 0.212993
NumberRealEstateLoansOrLines           -0.003932
NumberOfTime60-89DaysPastDueNotWorse    0.061537
NumberOfDependents                      0.194332
Outcome                                 1.000000
"""
def perf_credit(theta=np.zeros(10), n=20000, eps=0, seed=15, verbose=False):
    # Load data
    data = pd.read_csv('processed_credit_data.csv')

    # Sample n observations from the dataset
    sampled_data = data.sample(n=n, random_state=seed)
    
    if verbose:
        # Print mean of each feature for each class before modification
        means_class_minus1_before = sampled_data[sampled_data['Outcome'] == -1].mean()
        means_class_1_before = sampled_data[sampled_data['Outcome'] == 1].mean()
        
        print("Mean of features for class -1 before modification:\n", means_class_minus1_before)
        print("Mean of features for class 1 before modification:\n", means_class_1_before)
        

    # Create the vector u with eps in positions 2, 5, 7 and 0s elsewhere
    u = np.zeros(10)
    u[[0,2,5,6,8]] = eps

    # Modification of the features where the outcome is -1
    mask = sampled_data['Outcome'] == -1
    sampled_data.loc[mask, sampled_data.columns[:-1]] += u * theta

    # Split the data into features (X) and outcome (y)
    X = sampled_data.iloc[:, :-1]  # All columns except the last one (Outcome)
    y = sampled_data['Outcome']  # Only the Outcome column

    return X.values, y.values

def generate_synthetic_data( n,theta = np.zeros(2), Pi=np.zeros((2,2)), mu=-np.ones(1), scale=0.4, rng=np.random.RandomState()):

    # Calculate the number of samples per class
    n_per_class = n // 2

    # Generate n/2 samples from a Gaussian centered at 0
    X_class1 = rng.normal(loc=0, scale=scale, size=(n_per_class, theta.size))

    # Calculate the center for the second class
    center_class2 = mu + Pi @ theta

    # Generate n/2 samples from a Gaussian centered at mu + Pi theta
    X_class2 = rng.normal(loc=center_class2, scale=scale, size=(n_per_class, theta.size))

    # Create labels for the samples
    y_class1 = np.ones(n_per_class)   # Label 1 for the first class
    y_class2 = -np.ones(n_per_class)  # Label -1 for the second class

    # Combine the features and labels
    X = np.vstack((X_class1, X_class2))
    y = np.concatenate((y_class1, y_class2))

    indices = rng.permutation(n)
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":    # Example usage
    theta = np.array([0.5, -0.5, .3])
    n = 100
    X, y = generate_synthetic_data(n,theta, Pi = np.diag([0.2,0.6,0]))
    print("Features (first few rows):", X[:5])
    print("Labels (first few rows):", y[:5])

    X, y = perf_credit(theta=np.array([1,1,2,3,4,5,6,7,8,9]),n=1800, eps=1)
    print(X.head(n=10))