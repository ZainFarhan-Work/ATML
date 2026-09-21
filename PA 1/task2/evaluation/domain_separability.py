"""Can a probe still tell source from target in the frozen features?"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


@torch.no_grad()
def collect(features_module, loader, device):
    """Backbone features plus their domain tag (0 = source, 1 = target)."""
    features_module.eval()
    features, tags = [], []
    for images, domain in loader:
        features.append(features_module(images.to(device)).cpu())
        tags.append(domain)
    return torch.cat(features).numpy(), torch.cat(tags).numpy()


def separability(source_pack, target_pack, seed=SEED):
    """Held-out accuracy of a balanced logistic regression; 50% is chance."""
    x = np.concatenate([source_pack[0], target_pack[0]])
    y = np.concatenate([source_pack[1], target_pack[1]])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    probe = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    probe.fit(x_train, y_train)
    return probe.score(x_test, y_test) * 100
