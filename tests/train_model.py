import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from core.classifier import AIContentClassifier

def load_data(data_folder, files, expected_labels):
    texts = []
    labels = []
    for file in files:
        with open(os.path.join(data_folder, file), "r") as f:
            content = f.read()
            segments = content.split("------")
            for segment in segments:
                segment = segment.strip()
                if segment:
                    texts.append(segment)
                    labels.append(expected_labels[file])
    return texts, labels

def has_three_consecutive_low_pplx(pplx_list, threshold=40, count=3):
    consecutive = 0
    for pplx in pplx_list:
        if pplx < threshold:
            consecutive += 1
            if consecutive >= count:
                return 1
        else:
            consecutive = 0
    return 0

def extract_features(texts, classifier):
    features = []
    for text in texts:
        result = classifier.classify(text)
        low_pplx_flag = has_three_consecutive_low_pplx(list(result["pplx_map"].values()))
        features.append([
            result["average_pplx"],
            result["burstiness"],
            len(result["pplx_map"]),
            low_pplx_flag
        ])
    return features

def main():
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    files = ["human.txt", "gpt4o.txt", "gpt4o-mini.txt", "llama3-8b.txt"]
    expected_labels = {
        "human.txt": 0,
        "gpt4o.txt": 1,
        "gpt4o-mini.txt": 1,
        "llama3-8b.txt": 1,
    }

    texts, labels = load_data(data_folder, files, expected_labels)
    classifier = AIContentClassifier()
    features = extract_features(texts, classifier)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    joblib.dump(model, "ml_model.pkl")
    print("Trained model saved as ml_model.pkl")

if __name__ == "__main__":
    main()