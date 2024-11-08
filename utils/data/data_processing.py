import torch


def to_device(data, device):
    """Moves data to the configured device (CPU/GPU).

    Args:
        data: Input data (can be tensor, list, or dict)

    Returns:
        Data moved to appropriate device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    else:
        return data
