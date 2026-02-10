from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/iris_model.joblib")


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    joblib.dump(
        {
            "model": model,
            "target_names": iris.target_names.tolist(),
            "feature_names": iris.feature_names,
        },
        MODEL_PATH,
    )

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()