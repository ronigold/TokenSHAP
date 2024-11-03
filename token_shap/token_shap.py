from itertools import combinations
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import numpy as np
import random
import colorsys
import os

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

    # Define endpoint based on whether it's a chat or a single generate request
    api_endpoint = f"{api_url}/api/{'chat' if messages else 'generate'}"
    data = {'model': model, 'stream': stream}

    # Populate the request data
    if messages:
        data['messages'] = messages
    elif prompt:
        data['prompt'] = prompt
        if image_path:
            data['images'] = [encode_image_to_base64(image_path)]
    else:
        raise ValueError("Either messages for chat or a prompt for generate must be provided.")

    # Send request
    response = requests.post(api_endpoint, json=data, stream=stream)
    if response.status_code != 200:
        output_handler(f"Failed to retrieve data: {response.status_code}")
        return ""

    # Handle streaming or non-streaming responses
    full_response = ""
    if stream:
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
                full_response += response_part
                output_handler(response_part)
                if json_response.get('done', False):
                    break
    else:
        json_response = response.json()
        response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
        full_response += response_part
        output_handler(response_part)

    return full_response, response

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

# Model abstraction
class Model:
    def generate(self, prompt):
        raise NotImplementedError

class OllamaModel(Model):
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.api_url = api_url

    def generate(self, prompt):
        text_response, _ = interact_with_ollama(
            model=self.model_name,
            prompt=prompt,
            api_url=self.api_url,
            output_handler=lambda o: o
        )
        return text_response

class LocalModel(Model):
    def __init__(self, model_name_or_path, max_new_tokens=100, temperature=0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        input_length = input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()

class Splitter:
    def split(self, prompt):
        raise NotImplementedError

    def join(self, tokens):
        raise NotImplementedError

class StringSplitter(Splitter):
    def __init__(self, split_pattern = ' '):
        self.split_pattern = split_pattern
    
    def split(self, prompt):
        return re.split(self.split_pattern, prompt.strip())
    
    def join(self, tokens):
        return ' '.join(tokens)

class TokenizerSplitter(Splitter):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split(self, prompt):
        return self.tokenizer.tokenize(prompt)

    def join(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

# Refactored TokenSHAP class
class TokenSHAP:
    def __init__(self, model, splitter, debug=False):
        self.model = model
        self.splitter = splitter
        self.debug = debug  # Add debug mode

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def _calculate_baseline(self, prompt):
        baseline_text = self.model.generate(prompt)
        return baseline_text

    def _generate_random_combinations(self, samples, k, exclude_combinations_set):
        n = len(samples)
        sampled_combinations_set = set()
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))
        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_token_combination(self, prompt, sampling_ratio):
        samples = self.splitter.split(prompt)
        n = len(samples)
        self._debug_print(f"Number of samples (tokens): {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        num_total_combinations = 2 ** n - 1
        self._debug_print(f"Total possible combinations (excluding empty set): {num_total_combinations}")

        num_sampled_combinations = int(num_total_combinations * sampling_ratio)
        self._debug_print(f"Number of combinations to sample based on sampling ratio {sampling_ratio}: {num_sampled_combinations}")

        # Always include combinations missing one token
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        self._debug_print(f"Number of essential combinations (each missing one token): {len(essential_combinations)}")

        num_additional_samples = max(0, num_sampled_combinations - len(essential_combinations))
        self._debug_print(f"Number of additional combinations to sample: {num_additional_samples}")

        sampled_combinations = []
        if num_additional_samples > 0:
            sampled_combinations = self._generate_random_combinations(
                samples, num_additional_samples, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(sampled_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Combine essential and additional combinations
        all_combinations_to_process = essential_combinations + sampled_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations_to_process)}")

        prompt_responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations_to_process, desc="Processing combinations")):
            text = self.splitter.join(combination)
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations_to_process)}:")
            self._debug_print(f"Combination tokens: {combination}")
            self._debug_print(f"Token indexes: {indexes}")
            self._debug_print(f"Generated text: {text}")

            text_response = self.model.generate(text)
            self._debug_print(f"Received response for combination {idx + 1}")

            prompt_key = text + '_' + ','.join(str(index) for index in indexes)
            prompt_responses[prompt_key] = (text_response, indexes)

        self._debug_print("Completed processing all combinations.")
        return prompt_responses

    def _get_df_per_token_combination(self, prompt_responses, baseline_text):
        df = pd.DataFrame(
            [(prompt.split('_')[0], response[0], response[1])
             for prompt, response in prompt_responses.items()],
            columns=['Prompt', 'Response', 'Token_Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity(
            vectors[0].reshape(1, -1), vectors[1:]
        ).flatten()
        df["Cosine_Similarity"] = cosine_similarities

        return df

    def _calculate_shapley_values(self, df_per_token_combination, prompt):
        def normalize_shapley_values(shapley_values, power=1):
            min_value = min(shapley_values.values())
            shifted_values = {k: v - min_value for k, v in shapley_values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            normalized_values = {k: v / total for k, v in powered_values.items()}
            return normalized_values

        samples = self.splitter.split(prompt)
        n = len(samples)
        shapley_values = {}

        for i, sample in enumerate(samples, start=1):
            with_sample = np.average(
                df_per_token_combination[
                    df_per_token_combination["Token_Indexes"].apply(lambda x: i in x)
                ]["Cosine_Similarity"].values
            )
            without_sample = np.average(
                df_per_token_combination[
                    df_per_token_combination["Token_Indexes"].apply(lambda x: i not in x)
                ]["Cosine_Similarity"].values
            )

            shapley_values[sample + "_" + str(i)] = with_sample - without_sample

        return normalize_shapley_values(shapley_values)

    def print_colored_text(self):
        shapley_values = self.shapley_values
        min_value = min(shapley_values.values())
        max_value = max(shapley_values.values())

        def get_color(value):
            norm_value = (value - min_value) / (max_value - min_value)

            if norm_value < 0.5:
                r = int(255 * (norm_value * 2))
                g = int(255 * (norm_value * 2))
                b = 255
            else:
                r = 255
                g = int(255 * (2 - norm_value * 2))
                b = int(255 * (2 - norm_value * 2))

            return '#{:02x}{:02x}{:02x}'.format(r, g, b)

        for token, value in shapley_values.items():
            color = get_color(value)
            print(
                f"\033[38;2;{int(color[1:3], 16)};"
                f"{int(color[3:5], 16)};"
                f"{int(color[5:7], 16)}m"
                f"{get_text_before_last_underscore(token)}\033[0m",
                end=' '
            )
        print()

    def _get_color(self, value, shapley_values):
        norm_value = (value - min(shapley_values.values())) / (
            max(shapley_values.values()) - min(shapley_values.values())
        )
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))

    def plot_colored_text(self, new_line=False):
        num_items = len(self.shapley_values)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)

        for sample, value in self.shapley_values.items():
            color = self._get_color(value, self.shapley_values)
            if new_line:
                ax.text(
                    0.5, y_pos, get_text_before_last_underscore(sample), color=color, fontsize=20,
                    ha='center', va='center', transform=ax.transAxes
                )
                y_pos -= step
            else:
                ax.text(
                    x_pos, y_pos, get_text_before_last_underscore(sample), color=color, fontsize=20,
                    ha='left', va='center', transform=ax.transAxes
                )
                x_pos += 0.1

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(
                vmin=min(self.shapley_values.values()),
                vmax=max(self.shapley_values.values())
            )
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])
        cbar.set_label('Shapley Value', fontsize=12)

        plt.tight_layout()
        plt.show()

    def highlight_text_background(self):
        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

        def get_background_color(value):
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            return f"\033[48;2;{r};{g};{b}m"

        for token, value in self.shapley_values.items():
            background_color = get_background_color(value)
            reset_color = "\033[0m"
            print(f"{background_color}{get_text_before_last_underscore(token)}{reset_color}", end=' ')
        print()

    def analyze(self, prompt, sampling_ratio=0.0, print_highlight_text=False):
        # Clean the prompt to prevent empty tokens
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)

        self.baseline_text = self._calculate_baseline(prompt_cleaned)
        token_combinations_results = self._get_result_per_token_combination(
            prompt_cleaned, sampling_ratio
        )
        df_per_token_combination = self._get_df_per_token_combination(
            token_combinations_results, self.baseline_text
        )
        self.shapley_values = self._calculate_shapley_values(
            df_per_token_combination, prompt_cleaned
        )
        if print_highlight_text:
            self.highlight_text_background()

        return df_per_token_combination
