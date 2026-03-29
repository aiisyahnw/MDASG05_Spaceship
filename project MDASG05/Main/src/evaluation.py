from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate(test_data, pipeline):

    X_test, y_test = test_data

    predictions = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec  = recall_score(y_test, predictions)

    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print("-" * 30)

    return acc, prec, rec


if __name__ == "__main__":
    print("Jalankan")