import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import argparse


vocab_json_path = "D:/DDPM/Food-Vision/data/tokenizer_vocab.json"
merge_file_path = "D:/DDPM/Food-Vision/data/tokenizer_merges.txt"
model_file = "D:/DDPM/Food-Vision/data/v1-5-pruned-ema_only.ckpt"
dataset_path = "Dataset Path"


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Latent DDPM Conditional"
    args.epochs = 300
    args.batch_size = 8
    args.input_image_height = 64
    args.input_image_width = 64
    args.device = "cuda"
    args.seed = 42
    args.tokenizer = CLIPTokenizer(vocab_file=vocab_json_path, merges_file=merge_file_path)
    # Also known as negative prompt
    args.unconditional_prompt = dataset_path
    args.idle_device = "cpu"
    args.dataset_path = "Dataset Path"
    args.cfg_scale = 7
    args.lr = 3e-4


if __name__ == '__main__':
    launch()



