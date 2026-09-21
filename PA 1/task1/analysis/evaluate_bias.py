"""Evaluate shape/texture/color/spatial biases."""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

SEED = 6304
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256


def metrics(probabilities, labels):
    """Top-1 accuracy, macro-F1, and mean maximum confidence for one model."""
    confidence, predictions = probabilities.max(dim=1)
    return {
        "top1": (predictions == labels).float().mean().item() * 100,
        "macro_f1": f1_score(labels.numpy(), predictions.numpy(), average="macro") * 100,
        "mean_max_conf": confidence.mean().item(),
    }


def train_linear_head(train_features, train_labels, val_features, val_labels, seed=SEED):
    """Linear probe on frozen features; early stops on validation accuracy."""
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], int(train_labels.max()) + 1)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)

    best_accuracy, best_state, epochs_without_gain = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        head.train()
        order = torch.randperm(len(train_features), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = F.cross_entropy(head(train_features[batch]), train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            accuracy = (head(val_features).argmax(1) == val_labels).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"    early stop at epoch {epoch + 1}")
                break

    head.load_state_dict(best_state)
    print(f"    best val accuracy {best_accuracy * 100:.2f}%")
    return head, best_accuracy


@torch.no_grad()
def head_probabilities(head, features):
    """Softmax over the linear head's logits."""
    return F.softmax(head(features), dim=1)


@torch.no_grad()
def zero_shot_probabilities(image_features, text_features, logit_scale):
    """Softmax over CLIP's scaled image-text similarities."""
    return F.softmax(logit_scale * image_features @ text_features.T, dim=1)


def prediction_consistency(clean_probabilities, intervened_probabilities):
    """Percentage of images whose predicted class survives the intervention.

    Independent of correctness: an image wrong both before and after still counts
    as consistent, which is what makes this a measure of decision stability.
    """
    clean = clean_probabilities.argmax(dim=1)
    intervened = intervened_probabilities.argmax(dim=1)
    return (clean == intervened).float().mean().item() * 100
