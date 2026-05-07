import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast


# From evaluate_train_test_splits.py
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Use 'events' column for features, as it contains string-represented lists
    df['events'] = df['events'].apply(ast.literal_eval)  # Convert string to list
    X = np.array(df['events'].tolist())  # Convert list column to numpy array <--- FEATURES ARE EVENT FRAME NUMBERS

    # Use 'club' column as the target variable 'phase'  <--- TARGET IS 'club'
    y = df['club'].values
    return X, y


def evaluate_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }


def test_different_splits(csv_path):
    X, y = load_data(csv_path)
    split_ratios = {
        '70-30': 0.3,
        '90-10': 0.1,
        '60-40': 0.4,
        '80-20': 0.2
    }

    # Store raw metrics for calculating the best split
    raw_metrics_results = {}
    # Store formatted data for table display
    table_display_data = []

    print("Evaluating different train-test splits:\n")

    for name, test_size in split_ratios.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        metrics = evaluate_model(X_train, X_test, y_train, y_test)

        # Store raw metrics for finding the best split later
        raw_metrics_results[name] = metrics

        # Prepare data for the table: a dictionary for the current row
        current_row_for_table = {'Split': name}
        for metric_key, score_value in metrics.items():
            # Capitalize metric names for display (e.g., 'f1_score' -> 'F1 Score')
            display_metric_name = ' '.join(word.capitalize() for word in metric_key.split('_'))
            current_row_for_table[display_metric_name] = f"{score_value:.4f}"  # Format score
        table_display_data.append(current_row_for_table)

    # Create a pandas DataFrame from the collected data for table formatting
    metrics_summary_df = pd.DataFrame(table_display_data)

    # Define the desired column order for the output table
    column_display_order = ['Split', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    # Reorder DataFrame columns for consistent presentation
    metrics_summary_df = metrics_summary_df[column_display_order]

    # Print the DataFrame as a string, which gives a nice table format
    # index=False removes the DataFrame's default numerical index from the output
    print(metrics_summary_df.to_string(index=False))
    print()  # Add a blank line for better separation

    # Find the best split by highest F1 score using the raw metrics
    best_split_name = max(raw_metrics_results,
                          key=lambda k: raw_metrics_results[k]['f1_score'])
    best_f1_score = raw_metrics_results[best_split_name]['f1_score']

    print(f"✅ Best performing split: {best_split_name} with F1 Score = {best_f1_score:.4f}")


if __name__ == "__main__":
    csv_path = "GolfDB.csv"  # File is in the same directory
    test_different_splits(csv_path)