import os
import copy
import numpy as np
import torch
from torch.nn import functional as f
import model_loader_for_train
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import logging
from utils import *
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion, EMA
from ddpm_for_train import DDPMSampler

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# def get_data(args):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
#         torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     return dataloader


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


def get_time_embedding(timestep):
    # Shape: (160,)
    frequencies = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * frequencies[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def sampling(diffusion_model, conditional_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale):
    time_steps = tqdm(sampler.time_steps)
    # (Batch_Size, 4, Latents_Height, Latents_Width)
    latents = torch.randn(latents_shape, generator=generator, device=device)
    logging.info(f"Sampling {latents_shape[0]} new images....")
    diffusion_model.eval()
    with torch.no_grad():
        for i, timestep in enumerate(time_steps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            if cfg_scale:
                # model_output is the predicted noise for the unconditional context
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                unconditional_model_output = diffusion_model(latents, unconditional_context, time_embedding)

                # model_output is the predicted noise for the conditional context
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                conditional_model_output = diffusion_model(latents, conditional_context, time_embedding)

                model_output = cfg_scale * (conditional_model_output - unconditional_model_output) + unconditional_model_output
            else:
                # model_output is the predicted noise for the conditional context
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                model_output = diffusion_model(latents, conditional_context, time_embedding)

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)
    diffusion_model.train()
    return latents


def train(args):
    setup_logging(args.run_name)
    # Input declaration
    batch_size = args.batch_size
    input_image_height = args.input_image_height
    input_image_width = args.input_image_width
    device = args.device
    seed = args.seed
    tokenizer = args.tokenizer
    unconditional_prompt = args.unconditional_prompt
    idle_device = args.idle_device
    dataloader = get_data(args)
    diffusion = Diffusion().to(device)
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    def dump_to_idle_device(data_model):
        data_model.to(idle_device)

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    sampler = DDPMSampler(generator)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    length_of_data_loader = len(dataloader)
    ema = EMA(0.995)
    ema_diffusion = copy.deepcopy(diffusion).eval().requires_grad_(False)
    # Load the Pre-trained VAE and CLIP Encoder
    model_file = args.model_file

    current_prompt = []
    latents_width = input_image_width // 8
    latents_height = input_image_height // 8
    latents_shape = (batch_size, 4, latents_height, latents_width)
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        progress_bar = tqdm(dataloader)
        loss_per_epoch_list = []
        for i, (images, ingredients_prompt) in enumerate(progress_bar):

            # Get Time embeddings
            time_embedding = sampler.set_inference_time_steps(images.shape[0]).to(device)

            # Get Latent Tensor From VAE Encoder
            models = model_loader_for_train.preload_models_from_standard_weights(model_file, device, "encoder")
            encoder = models["current_load"]
            # encoder.to(device)
            latents_tensor = convert_image_tensor_to_latent_tensor(images, encoder, input_image_height, input_image_width, generator, device, latents_shape)
            dump_to_idle_device(encoder)

            # Get Context/Prompt From CLIP Encoder
            models = model_loader_for_train.preload_models_from_standard_weights(model_file, device, "clip")
            clip = models["current_load"]
            # clip.to(device)
            if np.random.random() < 0.1:
                # Convert into a list of length Seq_Len=77
                unconditional_tokens = tokenizer.batch_encode_plus([unconditional_prompt] * batch_size, padding="max_length", max_length=77).input_ids
                # (Batch_Size, Seq_Len)
                unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip(unconditional_tokens)
            else:
                # Convert into a list of length Seq_Len=77
                tokens = tokenizer.batch_encode_plus([ingredients_prompt], padding="max_length", max_length=77).input_ids
                # (Batch_Size, Seq_Len)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip(tokens)
            dump_to_idle_device(clip)

            # Add Noise in the original latents
            noisy_latents, actual_noise = sampler.add_noise(latents_tensor, time_embedding)

            # Predict the noise from diffusion model
            predicted_noise = diffusion(noisy_latents, context, time_embedding)

            # Calculate loss
            loss = mse(actual_noise, predicted_noise)
            loss_per_epoch_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_diffusion, diffusion)
            progress_bar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * length_of_data_loader + i)

            # Reassign current prompt
            current_prompt = ingredients_prompt
        average_loss = np.mean(np.array(loss_per_epoch_list))
        print(f"Average Epoch Loss: {average_loss} For Epoch Number: {epoch}")

        if epoch % 10 == 0:
            # Get Context/Prompt From CLIP Encoder
            models = model_loader_for_train.preload_models_from_standard_weights(model_file, device, "clip")
            clip = models["current_load"]
            # clip.to(device)

            # Generate unconditional context
            # Convert into a list of length Seq_Len=77
            unconditional_tokens = tokenizer.batch_encode_plus([unconditional_prompt] * batch_size, padding="max_length", max_length=77).input_ids
            # (Batch_Size, Seq_Len)
            unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            unconditional_context = clip(unconditional_tokens)

            # Generate conditional context
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus([current_prompt], padding="max_length", max_length=77).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            conditional_context = clip(tokens)
            dump_to_idle_device(clip)

            sampled_latents = sampling(diffusion, conditional_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)
            ema_sampled_latents = sampling(ema_diffusion, conditional_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)

            # Get the sampled images form the sampled latents using the VAE Decoder
            models = model_loader_for_train.preload_models_from_standard_weights(model_file, device, "decoder")
            decoder = models["current_load"]
            # decoder.to(device)
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
            sampled_images = decoder(sampled_latents)
            ema_sampled_images = decoder(ema_sampled_latents)
            dump_to_idle_device(decoder)

            # Save images
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

            # Save model checkpoint
            # Save diffusion model
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, os.path.join("models", args.run_name, f"check_point.pt"))

            # Save ema diffusion model
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema_diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, os.path.join("models", args.run_name, f"ema_check_point.pt"))



