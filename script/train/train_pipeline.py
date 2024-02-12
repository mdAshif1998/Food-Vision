import os
import copy
import numpy as np
import torch
import model_loader
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import logging
from utils import *
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion, EMA
from ddpm_for_train import DDPMSampler

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def convert_image_tensor_to_latent_tensor(input_image_tensor, vae_encoder, input_image_height, input_image_width, generator, device, latents_shape):

    input_image_tensor = input_image_tensor.resize((input_image_width, input_image_height))
    # (Height, Width, Channel)
    input_image_tensor = np.array(input_image_tensor)
    # (Height, Width, Channel) -> (Height, Width, Channel)
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
    # (Height, Width, Channel) -> (Height, Width, Channel)
    input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
    # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
    input_image_tensor = input_image_tensor.unsqueeze(0)
    # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
    input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

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

    models = model_loader.preload_models_from_standard_weights(model_file, device)
    current_prompt = []
    latents_width = input_image_width // 8
    latents_height = input_image_height // 8
    latents_shape = (1, 4, latents_height, latents_width)
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, ingredients_prompt) in enumerate(pbar):

            # Get Time embeddings
            time_embedding = sampler.set_inference_time_steps(images.shape[0]).to(device)

            # Get Latent Tensor From VAE Encoder
            encoder = models["encoder"]
            encoder.to(device)
            latents_tensor = convert_image_tensor_to_latent_tensor(images, encoder, input_image_height, input_image_width, generator, device, latents_shape)
            dump_to_idle_device(encoder)

            # Get Context/Prompt From CLIP Encoder
            clip = models["clip"]
            clip.to(device)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_diffusion, diffusion)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * length_of_data_loader + i)

            # Reassign current prompt
            current_prompt = ingredients_prompt

        if epoch % 10 == 0:
            # Get Context/Prompt From CLIP Encoder
            clip = models["clip"]
            clip.to(device)

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
            sampled_images = sampling(diffusion, conditional_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)
            ema_sampled_images = sampling(ema_diffusion, conditional_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)

            # Save images
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

            # Save model checkpoint
            torch.save(diffusion.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_diffusion.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


