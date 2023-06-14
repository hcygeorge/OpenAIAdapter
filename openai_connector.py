import openai
import random
import time
import os

class OpenAIConnector():
    """
    Initializes an instance of the OpenAIConnector class.

    Args:
        configs (dict): Configuration parameters for the adapter.
            api_key (str, optional): The API key used for authentication. Either `api_key` or `api_key_env`
                should be provided.
            api_key_env (str, optional): The environment variable name for the API key. If provided,
                the API key will be obtained from the environment variable.
            api_type (str, optional): The type of API to access. Defaults to 'open_ai'.
                                    Possible values: 'open_ai', 'azure'.
            azure_source_name (str, optional): The name of the Azure resource for the Azure API.
                                            Required when api_type is set to 'azure'.
            azure_api_version (str, optional): The version of the Azure API.
                                            Required when api_type is set to 'azure'.

    """
    def __init__(self, configs):
        openai.api_key = configs.get('api_key') or os.getenv(configs.get('api_key_env'))

        if configs.get('api_type') == 'azure':
            openai.api_type = 'azure'
            openai.api_base = f"https://{configs.get('azure_source_name')}.openai.azure.com/"
            openai.api_version = configs.get('azure_api_version')

        self.input_tokens = 0
        self.output_tokens = 0

    def _retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 200,
        errors: tuple = (openai.error.RateLimitError, openai.error.APIConnectionError),
    ):
        """
        Decorator function that retries the OpenAI API with exponential backoff when an exception is raised.

        Args:
            func: The function to decorate and retry.
            initial_delay (float, optional): The initial delay in seconds before the first retry. Defaults to 1.
            exponential_base (float, optional): The base value for exponential backoff calculation. Defaults to 2.
            jitter (bool, optional): Whether to apply jitter to the delay time. Defaults to True.
            max_retries (int, optional): The maximum number of retries. Defaults to 200.
            errors (tuple, optional): The tuple of specific errors to catch and retry on. 
                Defaults to (openai.error.RateLimitError,).

        Returns:
            The decorated function that performs the retries with exponential backoff.

        Raises:
            Exception: When the maximum number of retries is exceeded or an unhandled exception is raised.

        """
        def wrapper(self, *args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(self, *args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1
                    print(f'Number of API retries: {num_retries}')

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper


    @_retry_with_exponential_backoff
    def get_embedding(self, text, model_or_deployment):
        """
        Retrieves an embedding for the given text using OpenAI's text embedding API.
        @see https://platform.openai.com/docs/guides/embeddings/use-cases

        Args:
            text (str): The input text for which to retrieve the embedding.
            model_or_deployment (str): The name of the model or deployment to use for embedding generation.

        Returns:
            list: A list containing the embedding for the given text.

        Raises:
            OpenAIException: If an error occurs during the embedding retrieval process.
        """
        if openai.api_type == 'azure':
            return openai.Embedding.create(input=text, engine=model_or_deployment)['data'][0]['embedding']
        else:
            return openai.Embedding.create(input=text, model=model_or_deployment)['data'][0]['embedding']

    @_retry_with_exponential_backoff
    def ask_chatbot(self, messages, params):
        """
        Call the ChatGPT model with the given messages and optional keyword arguments.
        @see https://platform.openai.com/docs/guides/gpt/chat-completions-api
        @see https://platform.openai.com/docs/api-reference/chat/create

        Args:
            messages (list of string): A list of message discribing the conversation as far.
            params (dict):
                If openai.api_type is 'azure', the params should be in the format:
                {
                    'engine': 'your deployment name',
                    'max_tokens': 800,
                    'top_p': 1
                }

                If openai.api_type is 'openai', the params should be in the format:
                {
                    'model': 'your model name',
                    'max_tokens': 800,
                    'top_p': 1
                }

        Returns 
            OpenAIObject: the response from the ChatCompletion.create API call,
                it can be treated like a python dictionary.
        """
        response = openai.ChatCompletion.create(messages=messages, **params)
        if response:
            self.add_tokens(response)
        return response
    
    def get_answer(self, response):
        """
        Extracts the answer from the API response received from ChatGPT.

        Args:
            response (OpenAIObject): The response received from the ChatCompletion.create API.

        Returns:
            str: The extracted answer from the response.
                You can also access the answer by key, like response['choices'][0]['message']['content'].
        Raises:
            KeyError: If the answer cannot be extracted from the response.
        """
        return response.choices[0].message.content

    def add_tokens(self, response):
        """
        Add tokens from a completion response to the input and output token counts.

        Args:
            response (OpenAIObject): The response from ChatCompletion.create API.

        Returns:
            None
        """
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        
    def bill(self, input_price, output_price):
        """
        Calculate the accumulated cost(USD) of input and output tokens based on the provided pricing.

        Args:
            input_price (float): The input price(USD) per 1000 tokens.
            output_price (float): The output price(USD) per 1000 tokens.

        Returns:
            float: The accumulated cost(USD) of input and output tokens.
            
        Note:
            The actual pricing for OpenAI/Azure ChatCompletion services may vary.
                It is recommended to check the latest pricing information
                on the website before using this method.
                @see https://openai.com/pricing
                @see https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
        """
        input_cost = self.input_tokens * (input_price/1000)
        output_cost = self.output_tokens * (output_price/1000)
        accumulated_cost = input_cost + output_cost
        return accumulated_cost