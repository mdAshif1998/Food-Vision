from clip import CLIP
from encoder import VAEEncoder
from decoder import VAEDecoder
import torch
import gc
import model_converter


def get_model_state_dict(ckpt_path, device):
    return model_converter.load_from_standard_weights(ckpt_path, device)


def preload_models_from_standard_weights(model_ckpt_dict, device, util_network):

    if util_network == "encoder":
        current_load = VAEEncoder().to(device)
        current_load.load_state_dict(model_ckpt_dict['encoder'], strict=True)

    elif util_network == "decoder":
        current_load = VAEDecoder().to(device)
        current_load.load_state_dict(model_ckpt_dict['decoder'], strict=True)
    else:
        current_load = CLIP().to(device)
        current_load.load_state_dict(model_ckpt_dict['clip'], strict=True)

    return current_load





