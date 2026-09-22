"""Which unknowns slip through as confident known-class predictions."""

import numpy as np
import pandas as pd


def accepted_unknown_classes(scores, tau, fine_labels, names, predicted, known_names, top=5):
    """The unknown classes most often accepted, and what they are called."""
    accepted = np.asarray(scores) <= tau
    if not accepted.any():
        return {}
    rows = pd.DataFrame({
        "unknown_class": [names.get(int(l), str(int(l))) for l in np.asarray(fine_labels)[accepted]],
        "called": [known_names[int(p)] for p in np.asarray(predicted)[accepted]],
    })
    counts = rows.groupby(["unknown_class", "called"]).size().sort_values(ascending=False)
    return {f"{a} -> {b}": int(n) for (a, b), n in counts.head(top).items()}
