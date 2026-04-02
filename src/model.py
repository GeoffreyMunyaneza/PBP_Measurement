"""
U-Net model factory for BPD heatmap regression.

Architecture (Collins et al. 2026 best experiment):
    Encoder : ResNeXt101-32x8d pretrained on ImageNet
    Decoder : standard U-Net decoder
    Output  : 3 channels (left, right, center heatmaps)
    Input   : 3-channel image  (grayscale replicated)

The model outputs raw logits; sigmoid is applied during inference.
MSE loss is computed on raw outputs vs. [0, 1] Gaussian targets —
this is numerically equivalent to training with sigmoid because the
MSE gradient drives activations into the Gaussian range naturally.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(
    encoder_name: str = "resnext101_32x8d",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_heatmaps: int = 3,
) -> nn.Module:
    """
    Instantiate a U-Net with a ResNeXt101-32x8d encoder.

    Args:
        encoder_name    : smp encoder identifier.
        encoder_weights : Pre-training source ("imagenet" or None).
        in_channels     : Input image channels (3 for grayscale-replicated US).
        num_heatmaps    : Number of output heatmap channels (3: left, right, center).

    Returns:
        nn.Module ready for training or inference.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_heatmaps,
        activation=None,          # raw logits; sigmoid applied post-hoc for inference
        decoder_use_batchnorm=True,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, dict]:
    """
    Load model weights from a checkpoint saved by train.py.

    Returns:
        (model, checkpoint_dict)  — model is ready for inference on `device`.
    """
    checkpoint = torch.load(path, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
