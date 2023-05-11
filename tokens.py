# import openai
# import os

# openai.api_key = os.getenv('OPENAI_API_KEY')

import tiktoken

# encoding = tiktoken.get_encoding('p50k_base')

with open('msft10k.txt', 'r') as file:
    text = file.read()

# encoding.encode(text)

# num_tokens = len(encoding)

# print(num_tokens)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(text, 'p50k_base'))