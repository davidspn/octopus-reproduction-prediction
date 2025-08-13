import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, balanced_accuracy_score

def time_series_split(df, train_size_ratio=0.6):
    """Splits time series data into training and testing sets."""
    split_index = int(len(df) * train_size_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    
    y_train = train_data['Puestas']
    X_train = train_data.drop('Puestas', axis=1)
    
    y_test = test_data['Puestas']
    X_test = test_data.drop('Puestas', axis=1)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains the decision tree model and evaluates its performance."""
    model = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy Score on Test Set: {balanced_acc:.2%}\n")
    
    return model, y_pred

def plot_predictions(X_train, y_train, X_test, y_test, y_pred):
    """Plots the training data, actual test data, and predictions."""
    plt.figure(figsize=(20, 8))
    plt.step(X_train.index, y_train, label='Training Data (Actual)', color='tab:blue', where='mid')
    plt.step(X_test.index, y_test, label='Test Data (Actual)', color='gray', where='mid')
    plt.step(X_test.index, y_pred, label='Test Data (Predicted)', color='red', linestyle='--', where='mid')
    
    plt.title('Octopus Egg-Laying Prediction vs. Actual Events')
    plt.xlabel('Date')
    plt.ylabel('Egg Laying Event (1=Yes, 0=No)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_decision_tree(model, feature_names):
    """Plots the trained decision tree for interpretation."""
    plt.figure(figsize=(20, 12))
    plot_tree(model, 
              filled=True, 
              feature_names=feature_names, 
              class_names=['No Laying', 'Laying'],
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree for Octopus Egg-Laying Prediction")
    plt.show()
