import joblib
import os
import argparse

def load_trained_model(model_path='models/logistic_model.pkl', 
                      vectorizer_path='models/vectorizer.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_email(email_text, model, vectorizer):
    email_features = vectorizer.transform([email_text])
    prediction = model.predict(email_features)[0]
    probabilities = model.predict_proba(email_features)[0]
    return prediction, probabilities

def format_prediction_output(prediction, probabilities, email_text):
    print("\n" + "="*70)
    print("SPAM CLASSIFICATION RESULT")
    print("="*70)
    print(f"\nEmail Text:")
    print("-"*70)
    print(email_text[:200] + "..." if len(email_text) > 200 else email_text)
    print("-"*70)
    
    label = "SPAM" if prediction == 1 else "LEGITIMATE"
    confidence = probabilities[prediction]
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nProbability Distribution:")
    print(f"  Legitimate (0): {probabilities[0]*100:.2f}%")
    print(f"  Spam (1):       {probabilities[1]*100:.2f}%")
    print("="*70 + "\n")
    
    return label, confidence

def interactive_mode():
    print("="*70)
    print("SPAM EMAIL CLASSIFIER - INTERACTIVE MODE")
    print("="*70)
    print("\nLoading model...")
    
    try:
        model, vectorizer = load_trained_model()
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first by running:")
        print("  python logistic_regression_train.py")
        return
    
    print("\nInstructions:")
    print("  - Enter your email text (press Enter twice when done)")
    print("  - Type 'quit' or 'exit' to stop")
    print("="*70)
    
    while True:
        print("\n📧 Enter email text:")
        lines = []
        empty_count = 0
        
        while empty_count < 2:
            line = input()
            if line.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                return
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        
        email_text = "\n".join(lines).strip()
        
        if not email_text:
            print("⚠️  No text entered. Please try again.")
            continue
        
        prediction, probabilities = predict_email(email_text, model, vectorizer)
        format_prediction_output(prediction, probabilities, email_text)

def main():
    parser = argparse.ArgumentParser(
        description='Classify emails as spam or legitimate using Logistic Regression'
    )
    parser.add_argument('--text', type=str, help='Email text to classify')
    parser.add_argument('--file', type=str, help='Path to file containing email text')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if not args.text and not args.file and not args.interactive:
        args.interactive = True
    
    if args.interactive:
        interactive_mode()
        return
    
    print("Loading model...")
    try:
        model, vectorizer = load_trained_model()
        print("Model loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first by running:")
        print("  python logistic_regression_train.py")
        return
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            email_text = f.read()
    elif args.text:
        email_text = args.text
    else:
        print("Error: Please provide either --text or --file argument")
        return
    
    prediction, probabilities = predict_email(email_text, model, vectorizer)
    format_prediction_output(prediction, probabilities, email_text)

if __name__ == '__main__':
    main()
