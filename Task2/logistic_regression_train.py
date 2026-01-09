import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath='train_data.csv'):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    print(f"Spam samples: {sum(df['label'] == 1)}")
    print(f"Legitimate samples: {sum(df['label'] == 0)}")
    return df['text'].values, df['label'].values

def preprocess_and_train(texts, labels, test_size=0.3, random_state=42):
    print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Number of features: {X_train_tfidf.shape[1]}")
    
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        C=1.0,
        solver='liblinear'
    )
    
    model.fit(X_train_tfidf, y_train)
    print("Training completed!")
    
    train_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['Legitimate', 'Spam']))
    
    coefficients = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    
    top_spam_indices = np.argsort(coefficients)[-20:]
    print("\nTop 20 features indicating SPAM:")
    for idx in reversed(top_spam_indices):
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    top_legit_indices = np.argsort(coefficients)[:20]
    print("\nTop 20 features indicating LEGITIMATE:")
    for idx in top_legit_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    return model, vectorizer, cm, test_accuracy, X_test, y_test, test_pred, coefficients, feature_names

def save_model(model, vectorizer, model_path='models/logistic_model.pkl', 
               vectorizer_path='models/vectorizer.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

def create_visualizations(cm, accuracy, coefficients, feature_names, 
                         y_test, labels_count):
    os.makedirs('visualizations', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate (0)', 'Spam (1)'],
                yticklabels=['Legitimate (0)', 'Spam (1)'])
    plt.title('Confusion Matrix Heatmap\nSpam Email Classification', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nVisualization 1 saved: visualizations/confusion_matrix_heatmap.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    categories = ['Legitimate Emails', 'Spam Emails']
    counts = [labels_count[0], labels_count[1]]
    colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Class Distribution in Training Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Emails', fontsize=12)
    plt.xlabel('Email Category', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    print("Visualization 2 saved: visualizations/class_distribution.png")
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    top_spam_indices = np.argsort(coefficients)[-15:]
    top_legit_indices = np.argsort(coefficients)[:15]
    
    top_features = np.concatenate([top_legit_indices, top_spam_indices])
    top_coefs = coefficients[top_features]
    top_names = feature_names[top_features]
    
    colors_feat = ['green' if c < 0 else 'red' for c in top_coefs]
    
    plt.barh(range(len(top_coefs)), top_coefs, color=colors_feat, alpha=0.7)
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    plt.title('Top 30 Features in Logistic Regression Model\n(Red: Spam Indicators, Green: Legitimate Indicators)', 
              fontsize=13, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Visualization 3 saved: visualizations/feature_importance.png")
    plt.close()

def main():
    print("="*60)
    print("SPAM EMAIL DETECTION - LOGISTIC REGRESSION")
    print("="*60)
    
    texts, labels = load_data('train_data.csv')
    labels_count = [sum(labels == 0), sum(labels == 1)]
    
    model, vectorizer, cm, accuracy, X_test, y_test, y_pred, coefficients, feature_names = \
        preprocess_and_train(texts, labels)
    
    save_model(model, vectorizer)
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    create_visualizations(cm, accuracy, coefficients, feature_names, 
                         y_test, labels_count)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModel Performance Summary:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0][0]}")
    print(f"    False Positives: {cm[0][1]}")
    print(f"    False Negatives: {cm[1][0]}")
    print(f"    True Positives:  {cm[1][1]}")
    print("\nFiles saved:")
    print("  - models/logistic_model.pkl")
    print("  - models/vectorizer.pkl")
    print("  - visualizations/confusion_matrix_heatmap.png")
    print("  - visualizations/class_distribution.png")
    print("  - visualizations/feature_importance.png")
    print("="*60)

if __name__ == '__main__':
    main()
