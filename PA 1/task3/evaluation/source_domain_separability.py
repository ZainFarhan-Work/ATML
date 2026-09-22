"""How much source-domain identity survives in the frozen representation.

Three-way probe (Photo / Art Painting / Cartoon); chance is 33.3%. Lower means
the observed sources are harder to tell apart - which is not the same as the
representation being better.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


@torch.no_grad()
def collect(features_module, loader, device, domain_id):
    """Frozen features for one domain, tagged with that domain's id."""
    features_module.eval()
    collected = []
    for images, _ in loader:
        collected.append(features_module(images.to(device)).cpu())
    features = torch.cat(collected).numpy()
    return features, np.full(len(features), domain_id)


def separability(packs, seed=SEED):
    """Held-out accuracy of a balanced multinomial probe; chance is 33.3%."""
    x = np.concatenate([features for features, _ in packs])
    y = np.concatenate([labels for _, labels in packs])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    probe = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    probe.fit(x_train, y_train)
    return probe.score(x_test, y_test) * 100
