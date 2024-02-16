from clip import CLIP
from encoder import VAEEncoder
from decoder import VAEDecoder

import model_converter


def preload_models_from_standard_weights(ckpt_path, device, util_network):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    if util_network == "encoder":
        current_load = VAEEncoder().to(device)
        current_load.load_state_dict(state_dict['encoder'], strict=True)
    elif util_network == "decoder":
        current_load = VAEDecoder().to(device)
        current_load.load_state_dict(state_dict['decoder'], strict=True)
    else:
        current_load = CLIP().to(device)
        current_load.load_state_dict(state_dict['clip'], strict=True)

    return {
        'current_load': current_load,

    }





