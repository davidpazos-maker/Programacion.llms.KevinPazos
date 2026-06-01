import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def permutation_importance_mean(X: pd.DataFrame, y: np.ndarray, n_repeats: int) -> np.ndarray:
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42
    )

    return result.importances_mean
