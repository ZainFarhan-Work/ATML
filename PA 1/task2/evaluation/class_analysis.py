"""Per-class target accuracy and the confusions behind the biggest changes."""

import numpy as np
import pandas as pd


def per_class_accuracy(predicted, actual, classes):
    """Recall per class, in percent."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return {
        name: float((predicted[actual == index] == index).mean() * 100)
        if (actual == index).any()
        else float("nan")
        for index, name in enumerate(classes)
    }


def confusions(predicted, actual, classes, class_name, top=3):
    """What a class's images get called instead, most common first."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    index = classes.index(class_name)
    wrong = predicted[(actual == index) & (predicted != index)]
    counts = pd.Series([classes[p] for p in wrong]).value_counts()
    return counts.head(top).to_dict()
