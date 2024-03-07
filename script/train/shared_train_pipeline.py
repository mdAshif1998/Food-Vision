import os
import copy
import shutil
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


def get_time_embedding(timestep):
    # Shape: (160,)
    frequencies = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * frequencies[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def get_time_embedding_during_training(timestep: torch.Tensor) -> torch.Tensor:
    # Shape: (160,)
    frequencies = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor(timestep.to('cpu'), dtype=torch.float32)[:, None] * frequencies[None]
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

    model_file = args.model_file
    custom_dataset = CustomDataset(args.latent_file, args.context_file)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=custom_collate)
    # dataloader = get_data(args)

    def dump_to_idle_device(data_model):
        del data_model
        torch.cuda.empty_cache()

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    sampler = DDPMSampler(generator)
    if os.path.isdir(os.path.join(os.getcwd(), "runs")):
        shutil.rmtree(os.path.join(os.getcwd(), "runs"))
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    length_of_data_loader = len(dataloader)
    ema = EMA(0.995)

    diffusion = Diffusion().to(device)
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.lr)
    ema_diffusion = copy.deepcopy(diffusion).eval().requires_grad_(False)
    mse = nn.MSELoss()

    current_context = []
    latents_width = input_image_width // 8
    latents_height = input_image_height // 8
    latents_shape = (batch_size, 4, latents_height, latents_width)
    pretrained_network_state_dict = model_loader_for_train.get_model_state_dict(model_file, "cpu")
    # Generate unconditional context
    with torch.no_grad():
        # Get Context/Prompt From CLIP Encoder
        current_load = model_loader_for_train.preload_models_from_standard_weights(pretrained_network_state_dict, "cpu", "clip")
        # Generate unconditional context
        # Convert into a list of length Seq_Len=77
        unconditional_tokens = tokenizer.batch_encode_plus([unconditional_prompt] * batch_size, padding="max_length", max_length=77).input_ids
        # (Batch_Size, Seq_Len)
        unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device="cpu")
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        unconditional_context = current_load(unconditional_tokens)
        # dump_to_idle_device(current_load)
    # Start the training loop
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        progress_bar = tqdm(dataloader)
        loss_per_epoch_list = []
        for i, (latents_tensor, context) in enumerate(progress_bar):
            # Transform
            latents_tensor = latents_tensor.float()
            context = context.float()
            context = context.to(device)
            latents_tensor = latents_tensor.to(device)
            # Get Time embeddings
            time_embedding = sampler.set_inference_time_steps(batch_size).to(device)

            # Add Noise in the original latents
            noisy_latents, actual_noise = sampler.add_noise(latents_tensor, time_embedding)

            # Predict the noise from diffusion model
            time_embedding = get_time_embedding_during_training(time_embedding).to(device)
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
            current_context = context
        average_loss = np.mean(np.array(loss_per_epoch_list))
        print(f"Average Epoch Loss: {average_loss} For Epoch Number: {epoch}")

        if epoch % 10 == 0:

            sampled_latents = sampling(diffusion, current_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)
            ema_sampled_latents = sampling(ema_diffusion, current_context, unconditional_context, sampler, device, generator, latents_shape, cfg_scale=args.cfg_scale)

            with torch.no_grad():
                # Get the sampled images form the sampled latents using the VAE Decoder
                current_load = model_loader_for_train.preload_models_from_standard_weights(pretrained_network_state_dict, "cpu", "decoder")
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
                sampled_images = current_load(sampled_latents)
                ema_sampled_images = current_load(ema_sampled_latents)

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



