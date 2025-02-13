import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import re
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract additional features from text"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = np.zeros((len(X), 5))
        
        for i, text in enumerate(X):
            features[i, 0] = len(str(text).split())  # word count
            features[i, 1] = bool(re.search(r'\d{1,2}/\d{1,2}|\d{4}', str(text)))  # has date
            features[i, 2] = bool(re.search(r'urgent|asap|immediate|critical', str(text).lower()))  # urgent words
            features[i, 3] = bool(re.search(r'meeting|call|conference', str(text).lower()))  # meeting words
            features[i, 4] = bool(re.search(r'deadline|due|by|until', str(text).lower()))  # deadline words
            
        return features

def train_improved_model(file_path, model_save_path='improved_priority_classifier.joblib'):
    """
    Train an improved Random Forest classifier for task priority prediction.
    
    Parameters:
    file_path (str): Path to the processed CSV file
    model_save_path (str): Path where the trained model will be saved
    
    Returns:
    tuple: (trained_pipeline, results_dict)
    """
    # Load and prepare data
    print("Loading and preparing data...")
    df = pd.read_csv(file_path)
    
    # Check class balance
    print("\nClass distribution:")
    print(df['priority'].value_counts(normalize=True))
    
    # Create feature union pipeline
    feature_union = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )),
        ('text_features', TextFeatureExtractor())
    ])
    
    # Create main pipeline
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 20, 30, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Prepare features and target
    X = df['task']
    y = df['priority']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Perform grid search
    print("\nPerforming grid search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print("\nBest parameters:", results['best_params'])
    print("\nBest cross-validation score:", results['best_score'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=sorted(df['priority'].unique()),
                yticklabels=sorted(df['priority'].unique()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Priority')
    plt.xlabel('Predicted Priority')
    plt.show()
    
    # Save the model
    joblib.dump(best_model, model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Feature importance analysis
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': ['word_count', 'has_date', 'has_urgent', 'has_meeting', 'has_deadline'],
            'importance': best_model.named_steps['clf'].feature_importances_[-5:]
        })
        print("\nCustom Feature Importance:")
        print(feature_importance.sort_values('importance', ascending=False))
    
    return best_model, results

def evaluate_model_on_examples(model, examples):
    """
    Evaluate model on specific examples to check its behavior.
    
    Parameters:
    model: Trained pipeline
    examples (list): List of example tasks
    """
    predictions = model.predict(examples)
    probabilities = model.predict_proba(examples)
    
    print("\nModel evaluation on specific examples:")
    for task, pred, prob in zip(examples, predictions, probabilities):
        print(f"\nTask: {task}")
        print(f"Predicted Priority: {pred}")
        print(f"Confidence: {max(prob)*100:.2f}%")

# Example usage
if __name__ == "__main__":
    file_path = "processed_tasks.csv"
    
    # Train improved model
    model, results = train_improved_model(file_path)
    
    # Test on specific examples
    test_examples = [
        "Maybe call Kapalik",
        "Urgent: Client presentation due tomorrow",
        "Check email when free",
        "Critical system update required immediately",
        "Schedule team lunch next week"
    ]
    
    evaluate_model_on_examples(model, test_examples)