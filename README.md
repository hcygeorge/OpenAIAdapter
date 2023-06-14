# OpenAIConnector
It is a high-level interface for accessing ChatGPT and other OpenAI models from both **Official OpenAI** and **Azure OpenAI Service** using python.

## Features

- Support both OpenAI and Azure
- Auto retry when an API error occurs
- Count tokens and costs during connecting

## Requirements
- python > 3.7

## Installation
```
pip install openai
```

## ChatCompletion API Usage

```python
from openai_connector import OpenAIConnector

# Create chatbot
configs = {
    'api_key_env': 'Evironment variable of api key',
    'api_key': 'api key(no need if you have set api_key_env)',
    'api_type': 'azure or openai',
    'azure_source_name': 'Azure Source Name(only needed when you use Azure)',
    'azure_api_version': 'Azure API Version(only needed when you use Azure)'
}
chatbot = OpenAIAdapter(configs)

# Set model parameters
params = {
'engine': 'Deployment Name(only needed when you use Azure)',
'model': 'Model Name(only needed when you use OpenAI)',
'max_tokens': 800,
'top_p': 1
}

# Input history messages into chatbot
messages = [
        {"role":"system","content":"你是有禮貌的聊天機器人"},
        {"role":"user","content":"加拿大人口主要集中在那些省分?請用繁體中文回答"},
        {"role":"assistant","content":"加拿大人口主要集中在安大略省、魁北克省和不列顛哥倫比亞省。"},
        {"role":"user","content":"這三個省大約有多少人?"}
    ]

response = chatbot.ask_chatbot(messages, params=params)
print(chatbot.get_answer(response))  # get string of answer

# Count tokens and costs
chatbot.input_tokens  # accumulated intput tokens
chatbot.output_tokens  # accumulated output tokens

MAX_COST_LIMIT = 100
accumulated_cost = chatbot.bill(input_price=0.002, output_price=0.002)  # accumulated cost(USD)
if accumulated_cost >= MAX_COST_LIMIT:
    print('Stop your program when the cost reaches MAX_COST_LIMIT.')
```

## Embedding Model Usage

```python
configs = {
    'api_key_env':'API KEY Environment Variable',
    'api_type': 'azure',
    'azure_source_name': 'Azure Source Name',
    'azure_api_version': 'Azure API Version'
}
embed = OpenAIAdapter(configs)

vector = embed.get_embedding(
    text="葉黃素膠囊20mg 60粒",
    model_or_deployment="OpenAI Model Name or Azure Deployment Name"
    )
print(vector[:4])
```



## Reference
https://platform.openai.com/docs/api-reference/chat/create

https://learn.microsoft.com/en-us/azure/cognitive-services/openai/

