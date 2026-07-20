import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-manual-runs")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 4, 6]
}

# Grid search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
preds = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, preds)

# Log to MLflow
with mlflow.start_run():
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_cv_score", grid.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.sklearn.log_model(best_model, "model")

print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("Test Accuracy:", test_accuracy)
