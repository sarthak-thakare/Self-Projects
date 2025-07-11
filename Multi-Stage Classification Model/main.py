import numpy as np
from sklearn.datasets import load_digits  # Change dataset here if needed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import ClassificationPipeline, StandardNNClassifier

def main():
    # === Load your dataset ===
    data = load_digits()  # You can change to load_iris(), load_wine(), etc.
    X = data.data
    y = data.target

    # === Extract input/output dimensions ===
    input_size = X.shape[1]
    output_size = len(np.unique(y))

    print(f"Dataset: {data.__class__.__name__}, Features: {input_size}, Classes: {output_size}")

    # === Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Standardize ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === Evaluate multi-stage SVM + FFNN pipeline for each class ===
    accuracies = {}

    for svm_target in np.unique(y):
        print(f"\n--- Training SVM+FFNN Pipeline (Target = {svm_target}) ---")
        pipeline = ClassificationPipeline(svm_target=svm_target, input_size=input_size, output_size=output_size)
        pipeline.fit(X_train, y_train)
        pipeline_preds = pipeline.predict(X_test)
        pipeline_accuracy = np.mean(pipeline_preds == y_test)
        accuracies[f"Pipeline (SVM target {svm_target})"] = pipeline_accuracy * 100
        print(f"Test Accuracy: {pipeline_accuracy * 100:.2f}%")

    # === Evaluate standard FFNN ===
    print(f"\n--- Training Standard FFNN ---")
    standard_nn = StandardNNClassifier(input_size=input_size, output_size=output_size)
    standard_nn.fit(X_train, y_train)
    standard_nn_preds = standard_nn.predict(X_test)
    standard_nn_accuracy = np.mean(standard_nn_preds == y_test)
    accuracies["Standard NN"] = standard_nn_accuracy * 100
    print(f"Standard NN Test Accuracy: {standard_nn_accuracy * 100:.2f}%")

    # === Summary ===
    print("\n=== Summary of Accuracies ===")
    for model_name, acc in accuracies.items():
        print(f"{model_name}: {acc:.2f}%")

if __name__ == "__main__":
    main()
