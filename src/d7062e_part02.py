import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


def get_train_features_and_labels(input_file):
    gesture_training_data = pd.read_csv(input_file)
    train_features = gesture_training_data.iloc[:, 0:240].values
    train_labels = gesture_training_data.iloc[:, 241].values
    return [train_features, train_labels]


def handle_missing_values(features):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(features)
    features = imputer.transform(features)
    return features


def perform_normalization_by_scaling(train, test):
    feature_scaler = StandardScaler()
    normalized_train = feature_scaler.fit_transform(train)
    normalized_test = feature_scaler.fit_transform(test)
    return [normalized_train, normalized_test]


def train_decision_tree_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('###################################### DECISION TREE ALGORITHM ######################################')
    # Train the decision tree algorithm
    dt_reg = DecisionTreeClassifier()
    dt_reg.fit(train_features, train_labels)

    # Make predictions
    predictions = dt_reg.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    print(comparison)

    # Calculate MAE, MSE and RMSE
    print(f'MAE {metrics.mean_absolute_error(test_labels, predictions)}')
    print(f'MSE {metrics.mean_squared_error(test_labels, predictions)}')
    print(f'RMSE {np.sqrt(metrics.mean_squared_error(test_labels, predictions))}')
    print('-----------------------------------------------------------------------------------------------------')


def train_random_forest_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('###################################### RANDOM FOREST ALGORITHM ######################################')
    # Train the random forest algorithm
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_reg.fit(train_features, train_labels)

    # Make predictions
    predictions = rf_reg.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    print(comparison)

    # Calculate MAE, MSE and RMSE
    print(f'MAE {metrics.mean_absolute_error(test_labels, predictions)}')
    print(f'MSE {metrics.mean_squared_error(test_labels, predictions)}')
    print(f'RMSE {np.sqrt(metrics.mean_squared_error(test_labels, predictions))}')
    print('-----------------------------------------------------------------------------------------------------')


if __name__ == "__main__":
    # Load datasets for training and test and share features and labels
    INPUT_FILE_TRAIN = 'datasets/train-final.csv'
    INPUT_FILE_TEST = 'datasets/test-final.csv'
    train_list = get_train_features_and_labels(INPUT_FILE_TRAIN)
    test_list = get_train_features_and_labels(INPUT_FILE_TEST)

    # Get features and labels out of the training and test datasets
    train_features = train_list[0]
    train_labels = train_list[1]
    test_features = test_list[0]
    test_labels = test_list[1]

    # Handle missing values in the training and test dataset
    train_features = handle_missing_values(train_features)
    test_features = handle_missing_values(test_features)

    # Normalizing the data
    normalized_data = perform_normalization_by_scaling(
        train_features, test_features)
    train_features = normalized_data[0]
    test_features = normalized_data[1]

    # Run performance evaluation of decision tree
    train_decision_tree_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of decision tree
    train_random_forest_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)
