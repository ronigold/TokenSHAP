import requests
import json
import base64
from typing import Optional, Callable, Any
from os import getenv

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a base64-encoded string.
    
    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interact_with_ollama(
    prompt: Optional[str] = None, 
    messages: Optional[Any] = None,
    image_path: Optional[str] = None,
    model: str = 'openhermes2.5-mistral', 
    stream: bool = False, 
    output_handler: Callable[[str], None] = default_output_handler,
    api_url: Optional[str] = None
) -> str:
    """
    Sends a request to the Ollama API for chat or image generation tasks.
    
    Args:
    prompt (Optional[str]): Text prompt for generating content.
    messages (Optional[Any]): Structured chat messages for dialogues.
    image_path (Optional[str]): Path to an image for image-related operations.
    model (str): Model identifier for the API.
    stream (bool): If True, keeps the connection open for streaming responses.
    output_handler (Callable[[str], None]): Function to handle output messages.
    api_url (Optional[str]): API endpoint URL; if not provided, fetched from environment.

    Returns:
    str: Full response from the API.

    Raises:
    ValueError: If necessary parameters are not correctly provided or API URL is missing.
    """
    if not api_url:
        api_url = getenv('API_URL')
        if not api_url:
            raise ValueError('API_URL is not set. Provide it via the api_url variable or as an environment variable.')

    api_endpoint = f"{api_url}/{'chat' if messages else 'generate'}"
    data = {'model': model, 'stream': stream}

    if messages:
        data['messages'] = messages
    elif prompt:
        data['prompt'] = prompt
        if image_path:
            data['images'] = [encode_image_to_base64(image_path)]
    else:
        raise ValueError("Either messages for chat or a prompt for generate must be provided.")

    response = requests.post(api_endpoint, json=data, stream=stream)
    if response.status_code != 200:
        output_handler(f"Failed to retrieve data: {response.status_code}")
        return ""

    full_response = ""
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line.decode('utf-8'))
            response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
            full_response += response_part
            output_handler(response_part)
            if json_response.get('done', False):
                break

    return full_response, response