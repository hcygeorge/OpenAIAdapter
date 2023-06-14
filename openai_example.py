#!pip install openai
from commonlib.openai.openai_adapter import OpenAIAdapter

# ChatGPT usage
configs = {
    'api_key_env':'API KEY Environment Variable',
    'api_type': 'azure',
    'azure_source_name': 'Azure Source Name',
    'azure_api_version': 'Azure API Version'
}
chatbot = OpenAIAdapter(configs)

params = {
'engine': 'OpenAI Model Name or Azure Deployment Name',
'max_tokens': 800,
'top_p': 1
}

messages = [
        {"role":"system","content":"你是有禮貌的聊天機器人"},
        {"role":"user","content":"加拿大人口主要集中在那些省分?請用繁體中文回答"}
    ]

response = chatbot.ask_chatbot(messages, params=params)
print(chatbot.get_answer(response))

# Embedding model usage
configs = {
    'api_key_env':'API KEY Environment Variable',
    'api_type': 'azure',
    'azure_source_name': 'Azure Source Name',
    'azure_api_version': 'Azure API Version'
}
embed = OpenAIAdapter(configs)

vector = embed.get_embedding(
    text="羅技 MX Vertical 垂直滑鼠",
    model_or_deployment="OpenAI Model Name or Azure Deployment Name"
    )
print(vector[:4])