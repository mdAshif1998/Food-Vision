import os

import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path="D:/DDPM/Food-Vision/token/.env")
client = Groq(api_key=os.getenv("groq_token"))


def generate_food_prompt(raw_ingredient):
    food_prompt_prefix = """
    Give the list of comma separated common ingredients from this list of raw input list
    """
    return food_prompt_prefix + f''' "{raw_ingredient}"'''


def get_refined_common_ingredient(ingredient_prompt):
    prompt_list = [{'role': 'user', 'content': ingredient_prompt}]
    try:
        # completion = client.chat.completions.create(messages=prompt_list, model='mixtral-8x7b-32768', temperature=0.8)
        # Gemma-7b-it
        completion = client.chat.completions.create(messages=prompt_list, model='Gemma-7b-it', temperature=0.8)
        result = completion.choices[0].message.content
        return pd.Series([result, "Pass"], index=["common_ingredient", "status"])

    except Exception as result_generation_exception:
        return pd.Series(["", str(result_generation_exception)], index=["common_ingredient", "status"])


if __name__ == '__main__':
    raw_df = pd.read_excel("D:/DDPM/gemma_prompt/preprocessed_ingredient.xlsx", engine="openpyxl")
    raw_df = raw_df.head(10)
    tqdm.pandas()
    raw_df['food_prompt'] = raw_df["ingredients"].progress_apply(generate_food_prompt)
    tqdm.pandas()
    added_food_property = raw_df['food_prompt'].progress_apply(get_refined_common_ingredient)
    final_food_df = pd.concat([raw_df, added_food_property], axis=1)
    final_food_df.to_excel("D:/DDPM/gemma_prompt/added_common_ingredient.xlsx", engine="openpyxl", index=False)

