from train.utils import *
from tqdm import tqdm
import numpy as np
import argparse
from transformers import CLIPTokenizer
from train.model_loader_for_train import *


vocab_json_path = "D:/DDPM/Food-Vision/data/tokenizer_vocab.json"
merge_file_path = "D:/DDPM/Food-Vision/data/tokenizer_merges.txt"
model_file_path = "D:/DDPM/Food-Vision/data/v1-5-pruned-ema_only.ckpt"
# Dataset Path For Image
image_dataset_path = "E:/image"
# Excel path which contains the prompt and the corresponding image file name
excel_path = "E:/excel/preprocessed_ingredient.xlsx"


def convert_image_tensor_to_latent_tensor(input_image_tensor, vae_encoder, input_image_height, input_image_width, generator, device, latents_shape):

    # input_image_tensor = input_image_tensor.resize((input_image_width, input_image_height))
    # input_image_tensor = f.interpolate(input_image_tensor, size=(input_image_width, input_image_height), mode='bilinear', align_corners=False)
    # (Height, Width, Channel)
    input_image_tensor = np.array(input_image_tensor)
    # (Height, Width, Channel) -> (Height, Width, Channel)
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
    # (Height, Width, Channel) -> (Height, Width, Channel)
    input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
    # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
    # input_image_tensor = input_image_tensor.unsqueeze(0)
    # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
    # input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

    if device == 'cuda':
        input_image_tensor = input_image_tensor.to(device)
    # (Batch_Size, 4, Latents_Height, Latents_Width)
    encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
    # (Batch_Size, 4, Latents_Height, Latents_Width)
    latents = vae_encoder(input_image_tensor, encoder_noise)
    return latents


def store_context(args):
    device = args.device
    seed = args.seed
    tokenizer = args.tokenizer
    model_file = args.model_file
    dataloader = get_data(args)
    idle_device = args.idle_device
    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    def dump_to_idle_device(data_model):
        del data_model
        torch.cuda.empty_cache()

    pretrained_network_state_dict = get_model_state_dict(model_file)
    progress_bar = tqdm(dataloader)
    current_load = preload_models_from_standard_weights(pretrained_network_state_dict, device, "clip")
    all_context = np.random.random((1, 77, 768))
    for i, (images, ingredients_prompt, unique_index) in enumerate(progress_bar):
        # Get Context Representation From CLIP Encoder
        with torch.no_grad():

            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(ingredients_prompt, padding="max_length", max_length=77, truncation=True).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = current_load(tokens)
            # Delete the Current Loaded Network
            # dump_to_idle_device(current_load)
            all_context = np.concatenate((all_context, context.squeeze().to(idle_device).numpy()), axis=0)
    # Save the numpy arrays
    np.save('context.npy', all_context[1:])


def store_latent_tensor(args):
    batch_size = args.batch_size
    input_image_height = args.input_image_height
    input_image_width = args.input_image_width
    device = args.device
    idle_device = args.idle_device
    seed = args.seed
    model_file = args.model_file
    dataloader = get_data(args)

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    def dump_to_idle_device(data_model):
        del data_model
        torch.cuda.empty_cache()

    latents_width = input_image_width // 8
    latents_height = input_image_height // 8
    latents_shape = (batch_size, 4, latents_height, latents_width)
    pretrained_network_state_dict = get_model_state_dict(model_file)
    progress_bar = tqdm(dataloader)
    all_latent = np.random.random((1, 4, 64, 64))
    current_load = preload_models_from_standard_weights(pretrained_network_state_dict, device, "encoder")
    for i, (images, ingredients_prompt, unique_index) in enumerate(progress_bar):
        # Get Latent Tensor From VAE Encoder
        with torch.no_grad():
            latents_tensor = convert_image_tensor_to_latent_tensor(images, current_load, input_image_height, input_image_width, generator, device, latents_shape)
            # Delete the Current Loaded Network
            # dump_to_idle_device(current_load)
            all_latent = np.concatenate((all_latent, latents_tensor.squeeze().to(idle_device).numpy()), axis=0)

    # Save the numpy arrays
    np.save('latent_tensor.npy', all_latent[1:])


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Latent DDPM Conditional"
    args.epochs = 300
    args.batch_size = 8
    args.input_image_height = 512
    args.input_image_width = 512
    args.image_size = 512
    args.device = "cuda"
    args.seed = 42
    # Also known as negative prompt
    args.unconditional_prompt = ""
    args.idle_device = "cpu"
    args.tokenizer = CLIPTokenizer(vocab_file=vocab_json_path, merges_file=merge_file_path)
    args.image_dataset_path = image_dataset_path
    args.excel_path = excel_path
    args.cfg_scale = 7
    args.lr = 3e-4
    args.model_file = model_file_path
    args.dataset_set = 10000
    store_latent_tensor(args)
    store_context(args)


if __name__ == '__main__':
    launch()


