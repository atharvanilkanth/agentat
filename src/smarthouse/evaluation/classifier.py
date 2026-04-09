"""Optional classifier transfer experiment."""
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def extract_features(window: Dict[str, Any]) -> np.ndarray:
    """Extract a feature vector from a window for classification."""
    events_a = window.get("events_a_overlap", [])
    events_b = window.get("events_b_overlap", [])
    all_events = window.get("merged_events", [])

    features = [
        len(events_a),
        len(events_b),
        len(all_events),
        window.get("overlap_duration_seconds", 0),
        len(window.get("rooms_a", [])),
        len(window.get("rooms_b", [])),
        len(set(window.get("rooms_a", [])) & set(window.get("rooms_b", []))),
        float(bool(events_a)),
        float(bool(events_b)),
    ]
    return np.array(features, dtype=float)


def run_classifier_experiment(
    real_windows: List[Dict[str, Any]],
    baseline_windows: List[Dict[str, Any]],
    validated_windows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Train a simple classifier to distinguish real vs synthetic windows.
    Lower accuracy = more realistic synthetic data.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    results = {}

    for synth_name, synth_windows in [
        ("baseline", baseline_windows),
        ("validated", validated_windows),
    ]:
        if not real_windows or not synth_windows:
            results[synth_name] = {"error": "empty windows"}
            continue

        n = min(len(real_windows), len(synth_windows))
        X_real = np.array([extract_features(w) for w in real_windows[:n]])
        X_synth = np.array([extract_features(w) for w in synth_windows[:n]])
        y = np.array([1] * n + [0] * n)
        X = np.vstack([X_real, X_synth])

        # Handle NaN/Inf
        X = np.nan_to_num(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=min(3, n), scoring="accuracy")

        results[synth_name] = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "n_real": n,
            "n_synth": n,
        }
        logger.info(
            "Classifier (%s vs real): accuracy=%.3f +/- %.3f",
            synth_name, results[synth_name]["mean_accuracy"],
            results[synth_name]["std_accuracy"],
        )

    return results
