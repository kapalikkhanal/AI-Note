import json
import pandas as pd
import numpy as np

def preprocess_task_data(file_path):
    """
    Preprocess task dataset by converting categorical priorities to numerical values
    and performing basic data cleaning.
    
    Parameters:
    file_path (str): Path to the JSON file containing task data
    
    Returns:
    pandas.DataFrame: Preprocessed dataset
    """
    # Read JSON file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create priority mapping
    priority_map = {
        'High': 3,
        'Medium': 2,
        'Low': 1
    }
    
    # Convert priority to numerical values
    df['priority_numeric'] = df['priority'].map(priority_map)
    
    # Create one-hot encoded version of priority (optional)
    priority_dummies = pd.get_dummies(df['priority'], prefix='priority')
    df = pd.concat([df, priority_dummies], axis=1)
    
    # Basic statistics of the dataset
    stats = {
        'total_tasks': len(df),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'average_priority': df['priority_numeric'].mean(),
        'priority_stats': df['priority_numeric'].describe().to_dict()
    }
    
    print("\nDataset Statistics:")
    print(f"Total number of tasks: {stats['total_tasks']}")
    print("\nPriority Distribution:")
    for priority, count in stats['priority_distribution'].items():
        print(f"{priority}: {count} tasks ({(count/stats['total_tasks']*100):.1f}%)")
    print(f"\nAverage Priority Score: {stats['average_priority']:.2f}")
    
    return df, stats

# Example usage
if __name__ == "__main__":
    file_path = "datasets.json"
    
    # Process the data
    processed_df, statistics = preprocess_task_data(file_path)
    
    if processed_df is not None:
        print("\nFirst few rows of processed dataset:")
        print(processed_df.head())
        
        # Save processed dataset to CSV (optional)
        processed_df.to_csv('processed_tasks.csv', index=False)
        print("\nProcessed dataset saved to 'processed_tasks.csv'")