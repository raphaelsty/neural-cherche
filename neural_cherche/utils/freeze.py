from transformers import AutoModelForMaskedLM

__all__ = ["freeze_layers"]


def freeze_layers(
    model: AutoModelForMaskedLM, n_layers: int = 5
) -> AutoModelForMaskedLM:
    """Freeze layers before the last n_layers."""
    trainable_layers = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers += 1

    for index, (name, param) in enumerate(iterable=model.named_parameters()):
        if index < (trainable_layers - 5):
            param.requires_grad = False

    return model
