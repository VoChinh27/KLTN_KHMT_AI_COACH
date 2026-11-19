"""One approved change: squared error, same supervised-video mean reduction."""
from typing import Literal
from ..research.contract import Config as OriginalConfig


class Config(OriginalConfig):
    loss: Literal['mse_video_mean'] = 'mse_video_mean'


def descriptor():
    return dict(version='squat-count-mse-loss-v1', formula='mean((prediction-target)^2) over target_mask',
        reduction='mean_supervised_videos_per_batch', epoch_log='sample_count_weighted_batch_mean_during_updates',
        target_transform='none', count_weights='none', prediction_clipping='none',
        retained_gradient_norm_clip=1.0, input_transform='identity_original_raw58')


def count_loss(prediction, batch):
    import torch
    mask, target = batch['target_mask'], batch['target']
    if (prediction.ndim != 1 or target.shape != prediction.shape or mask.shape != prediction.shape
            or mask.dtype != torch.bool or not mask.any()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all()):
        raise ValueError('MSE requires finite predictions and supervised targets with aligned mask')
    result = torch.nn.functional.mse_loss(prediction[mask], target[mask], reduction='mean')
    if not torch.isfinite(result):
        raise ValueError('Non-finite MSE; no update permitted')
    return result
