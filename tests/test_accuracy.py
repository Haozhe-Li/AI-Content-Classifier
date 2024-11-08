import os
import asyncio
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.classifier import AIContentClassifier

@pytest.mark.asyncio
async def test_accuracy():
    detector = AIContentClassifier()
    data_folder = "./data"
    files = ["human.txt", "gpt4o.txt", "gpt4o-mini.txt", "llama3-8b.txt"]
    expected_labels = {
        "human.txt": 0,
        "gpt4o.txt": 1,
        "gpt4o-mini.txt": 1,
        "llama3-8b.txt": 1,
    }

    correct_predictions = 0
    total_predictions = 0
    model_accuracies = {file: {"correct": 0, "total": 0} for file in files}

    for file in files:
        print(f"Processing file: {file}")
        with open(os.path.join(data_folder, file), "r") as f:
            content = f.read()
            segments = content.split("------")
            for i, segment in enumerate(segments):
                segment = segment.strip()
                if segment:
                    result = await detector.classify(segment)
                    label = result["label"]
                    if label == expected_labels[file]:
                        correct_predictions += 1
                        model_accuracies[file]["correct"] += 1
                    else:
                        print(f"Incorrect prediction on {i + 1} of {file}, expected {expected_labels[file]} but got {label}. The likelihood score was {result['likelihood_score']}. Average PPLX was {result['average_pplx']}.")
                    model_accuracies[file]["total"] += 1
                    total_predictions += 1

    overall_accuracy = correct_predictions / total_predictions
    model_accuracies = {file: acc["correct"] / acc["total"] for file, acc in model_accuracies.items()}

    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    for model, accuracy in model_accuracies.items():
        print(f"Accuracy for {model}: {accuracy:.2f}")

    assert overall_accuracy >= 0.85, "Overall accuracy is below 85%"

if __name__ == "__main__":
    asyncio.run(test_accuracy())