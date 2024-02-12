
from transformers import CLIPTokenizer
import argparse
from train_pipeline import train

vocab_json_path = "D:/DDPM/Food-Vision/data/tokenizer_vocab.json"
merge_file_path = "D:/DDPM/Food-Vision/data/tokenizer_merges.txt"
model_file = "D:/DDPM/Food-Vision/data/v1-5-pruned-ema_only.ckpt"
# Dataset Path For Image
image_dataset_path = "E:/image"
# Excel path which contains the prompt and the corresponding image file name
excel_path = "E:/excel/preprocessed_ingredient.xlsx"


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Latent DDPM Conditional"
    args.epochs = 300
    args.batch_size = 8
    args.input_image_height = 64
    args.input_image_width = 64
    args.image_size = 64
    args.device = "cuda"
    args.seed = 42
    args.tokenizer = CLIPTokenizer(vocab_file=vocab_json_path, merges_file=merge_file_path)
    # Also known as negative prompt
    args.unconditional_prompt = ""
    args.idle_device = "cpu"
    args.dataset_path = image_dataset_path
    args.excel_path = excel_path
    args.cfg_scale = 7
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()



