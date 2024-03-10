import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv(dotenv_path="Provide Path Here")

prompt_for_common_ingredients_extraction = """
What is the list of common ingredients from this list of raw input list "[{'text': '1 lb Ground beef'}, {'text': '40 oz kidney beans'}, {'text': '1 cooked Jasmine white rice (or your favorite )'}, {'text': '1 medium onion'}, {'text': '4 large large bell peppers'}, {'text': '1 garlic powder'}, {'text': '1 onion powder'}, {'text': '1 can can diced tomatoes ( fiery garlic or your choice )'}, {'text': '24 oz spaghetti sauce (I used garden vegetable )'}, {'text': '1 shredded cheese (sharp cheddar and 6 cheese blend)'}]"

"""
client = Groq(api_key=os.getenv("groq_token"))

completion = client.chat.completions.create(
    messages=[{'role': 'user', 'content': prompt_for_common_ingredients_extraction}],
    model='mixtral-8x7b-32768'

)

result = completion.choices[0].message.content

print(result)


