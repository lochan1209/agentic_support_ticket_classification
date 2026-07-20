import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-manual-runs")


# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try a few parameter combinations manually
for n_estimators in [10, 50, 100]:
    for max_depth in [2, 4, 6]:
        with mlflow.start_run():
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(model, "model")

            print(
                f"n_estimators={n_estimators}, "
                f"max_depth={max_depth}, "
                f"accuracy={acc:.4f}"
            )
