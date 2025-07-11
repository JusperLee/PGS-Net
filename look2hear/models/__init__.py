from .SwiftNet import SwiftNet
from .av_convtasnet_causal import av_convtasnet_causal
from .av_sepformer_causal import AV_Sepformer_Causal
from .av_tfgridnetv3_separator_causal import av_TFGridNetV3_causal
from .av_dprnn_causal import AV_Dprnn_Causal
from .ctcnet_causal import CTCNet_Causal
from .rtfsnet_causal import RTFSNetCausal

__all__ = [
    "SwiftNet"
    "av_convtasnet_causal"
    "AV_Sepformer_Causal"
    "av_TFGridNetV3_causal"
    "AV_Dprnn_Causal"
    "CTCNet_Causal"
    "RTFSNetCausal"
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
