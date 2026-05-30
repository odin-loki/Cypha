from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def offline_regression_baselines(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict[str, float]]:
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    results: dict[str, dict[str, float]] = {}

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    results["dummy_mean"] = _regression_scores(y_test, dummy.predict(X_test))

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    results["knn_5"] = _regression_scores(y_test, knn.predict(X_test))

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    results["ridge"] = _regression_scores(y_test, ridge.predict(X_test))

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = _regression_scores(y_test, rf.predict(X_test))

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    results["gradient_boosting"] = _regression_scores(y_test, gbr.predict(X_test))

    return results


def offline_classification_baselines(X_train: np.ndarray, y_train, X_test: np.ndarray, y_test) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    n_classes = len(np.unique(y_train))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    results["dummy_majority"] = _classification_scores(y_test, dummy.predict(X_test))

    if n_classes < 2:
        for name in ("knn_5", "logistic_regression", "random_forest", "gradient_boosting"):
            results[name] = results["dummy_majority"]
        return results

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    results["knn_5"] = _classification_scores(y_test, knn.predict(X_test))

    logreg = LogisticRegression(max_iter=500, random_state=42)
    try:
        logreg.fit(X_train, y_train)
        results["logistic_regression"] = _classification_scores(y_test, logreg.predict(X_test))
    except ValueError:
        results["logistic_regression"] = results["dummy_majority"]

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = _classification_scores(y_test, rf.predict(X_test))

    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    results["gradient_boosting"] = _classification_scores(y_test, gbc.predict(X_test))

    return results


def sgd_online_regressor(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    reg = SGDRegressor(max_iter=1, learning_rate="constant", eta0=0.01, random_state=42)
    for x, y in zip(X_train, y_train):
        reg.partial_fit(x.reshape(1, -1), [y])
    return _regression_scores(y_test, reg.predict(X_test))


def sgd_online_classifier(X_train: np.ndarray, y_train, X_test: np.ndarray, y_test, classes=None) -> dict[str, float]:
    classes = np.unique(y_train) if classes is None else np.asarray(classes)
    if len(classes) < 2:
        majority = classes[0] if len(classes) else 0
        preds = np.full(len(y_test), majority)
        return _classification_scores(y_test, preds)
    clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="constant", eta0=0.01, random_state=42)
    for x, y in zip(X_train, y_train):
        clf.partial_fit(x.reshape(1, -1), [y], classes=classes)
    return _classification_scores(y_test, clf.predict(X_test))
