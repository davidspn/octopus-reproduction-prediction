from data_processing import create_final_dataframe, create_lag_features
from modeling import time_series_split, train_and_evaluate, plot_predictions, plot_decision_tree

def main():
    """Main function to run the entire analysis pipeline."""
    
    # 1. Data Processing and Feature Engineering
    print("Step 1: Processing data and creating features...")
    temp_path = 'data/temperaturapulpos.csv'
    laying_path = 'data/puestaspulpos.xlsx'
    
    df = create_final_dataframe(temp_path, laying_path)
    df_features = create_lag_features(df)
    
    # 2. Data Splitting
    print("Step 2: Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = time_series_split(df_features, train_size_ratio=0.5) # Using a 50/50 split
    
    # 3. Modeling and Evaluation
    print("Step 3: Training model and evaluating performance...")
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 4. Visualization
    print("Step 4: Generating visualizations...")
    plot_predictions(X_train, y_train, X_test, y_test, y_pred)
    plot_decision_tree(model, X_train.columns)
    
    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()
