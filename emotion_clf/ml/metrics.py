"""Metrics"""
from typing import Dict, List, Tuple

import numpy as np
import sklearn.metrics
import torch
from torch import Tensor
from scipy import linalg
import emotion_clf.ml.losses as ml_losses


def extract_first_metric(metrics: Dict[str, np.ndarray]) -> float:
    return float(list(metrics.values())[0].mean())


def _avg_sigmoid(outs: Tensor) -> Tensor:
    return torch.mean(torch.nn.Sigmoid()(outs), 2)


def averaged_sigmoid(outputs: List[Tensor],
                     targets: List[Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    probas = torch.cat(list(map(_avg_sigmoid, outputs))).cpu().numpy()
    targs = torch.cat(targets).cpu().numpy()
    return probas, targs


def roc_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.roc_auc_score(targets > 0.5, probas, average=None)


def pr_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.average_precision_score(targets > 0.5,
                                                   probas,
                                                   average=None)


def bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = torch.nn.BCELoss(reduction='none')
    targs = torch.Tensor(targets)
    probs = torch.Tensor(probas)
    return torch.mean(loss(probs, targs), 0).cpu().numpy()


def focal_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml_losses.BinaryFocalLossWithLogits(reduction='none')
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()


def weighted_bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml_losses.WeightedBCEWithLogitsLoss(reduction="none")
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()

def frechet_audio_distance(real_features, gen_features):

    if isinstance(real_features, torch.Tensor):
        real_features = real_features.detach().cpu().numpy()
    if isinstance(gen_features, torch.Tensor):
        gen_features = gen_features.detach().cpu().numpy()

    # Calculate mean and covariance
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    # Calculate FAD
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2*covmean)

    return fad
