from itertools import combinations
from transformers import AutoTokenizer
from ollama_interact import interact_with_ollama
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import yaml
import itertools
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import colors
import re 
import numpy as np
import random
import numpy as np
import colorsys
import os

def get_text_before_last_underscore(token):
    # Find the last occurrence of '_' and slice up to that position
    return token.rsplit('_', 1)[0]

class TokenSHAP:
    def __init__(self, model_name, ollama_api_url, tokenizer_path=None):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_name
        self.ollama_api = ollama_api_url 
        
    def _calculate_baseline(self, prompt):
        baseline_text, _ = interact_with_ollama(model=self.model_name, prompt=prompt, api_url=self.ollama_api, output_handler=lambda o: o)
        return baseline_text
    
    def _get_result_per_token_combination(self, prompt, splitter, sampling_ratio):
        if splitter is None:
            samples = self.tokenizer.tokenize(prompt)
            is_tokenized = True
        else:
            samples = prompt.split(splitter)
            is_tokenized = False
    
        if len(samples) > 100:
            print("warning: the samples size is greather than 100, execution will be slow")
        
        num_total_combinations = 2**len(samples) - 1
        num_sampled_combinations = int(num_total_combinations * sampling_ratio)
    
        # Always include combinations where only one token is omitted
        essential_combinations = []
        for i in range(len(samples)):
            combination = samples[:i] + samples[i+1:]
            essential_combinations.append((combination, [j+1 for j in range(len(samples)) if j != i]))
    
        # Generate all possible combinations
        all_combinations = [(samples, list(range(1, len(samples)+1)))]
        for r in range(1, len(samples)):
            all_combinations.extend([(list(comb), [i+1 for i, token in enumerate(samples) if token in comb]) 
                                     for comb in combinations(samples, r)])
    
        # Remove essential combinations from all combinations
        remaining_combinations = [comb for comb in all_combinations if comb not in essential_combinations]
    
        # Sample additional combinations
        num_additional_samples = max(1, num_sampled_combinations - len(essential_combinations))
        sampled_combinations = random.sample(remaining_combinations, 
                                             min(num_additional_samples, len(remaining_combinations)))
    
        # Combine essential and sampled combinations
        all_combinations_to_process = essential_combinations + sampled_combinations
    
        prompt_responses = {}
        for combination, indexes in all_combinations_to_process:
            if is_tokenized:
                text = self.tokenizer.convert_tokens_to_string(combination)
            else:
                text = splitter.join(combination)
    
            text_response, response = interact_with_ollama(
                model=self.model_name,
                prompt=text,
                api_url=self.ollama_api,
                stream=True, 
                output_handler=lambda o: o
            )
            prompt_responses[text + '_' + ','.join(str(index) for index in indexes)] = (text_response, indexes)
        
        return prompt_responses
    
    def _get_df_per_token_combination(self, prompt_responses, baseline_text):
        df = pd.DataFrame([(prompt.split('_')[0], response[0], response[1]) 
                           for prompt, response in prompt_responses.items()],
                          columns=['Prompt', 'Response', 'Token_Indexes'])
   
        all_texts = [baseline_text] + df["Response"].tolist()
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:]).flatten()
        df["Cosine_Similarity"] = cosine_similarities
        
        return df
    
    def _calculate_shapley_values(self, df_per_token_combination, prompt, splitter):
        def normalize_shapley_values(shapley_values, power=1):
            # Step 1: Shift values to make them all non-negative
            min_value = min(shapley_values.values())
            shifted_values = {k: v - min_value for k, v in shapley_values.items()}
            
            # Step 2: Apply power transformation to accentuate differences
            powered_values = {k: v**power for k, v in shifted_values.items()}
            
            # Step 3: Normalize to sum to 1
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1/len(powered_values) for k in powered_values}
            normalized_values = {k: v / total for k, v in powered_values.items()}
            
            return normalized_values
    
        if splitter is None:            
            tokens = self.tokenizer.tokenize(prompt)
            samples = [token.strip('Ġ') if token.startswith('Ġ') else token for token in tokens]
        else:
            samples = prompt.split(splitter)
            
        n = len(samples)
        shapley_values = {}
    
        for i, sample in enumerate(samples, start=1):  # Start enumeration from 1 to match token indexes
            with_sample = np.average(df_per_token_combination[df_per_token_combination["Token_Indexes"].apply(lambda x: i in x)]["Cosine_Similarity"].values)
            without_sample = np.average(df_per_token_combination[df_per_token_combination["Token_Indexes"].apply(lambda x: i not in x)]["Cosine_Similarity"].values)

            shapley_values[sample + "_" + str(i)] = with_sample - without_sample
    
        return normalize_shapley_values(shapley_values)
    
    def print_colored_text(self):
        shapley_values = self.shapley_values
        min_value = min(shapley_values.values())
        max_value = max(shapley_values.values())
        
        def get_color(value):
            # Normalize value to 0-1 range
            norm_value = (value - min_value) / (max_value - min_value)
            
            if norm_value < 0.5:
                # Blue to White
                r = int(255 * (norm_value * 2))
                g = int(255 * (norm_value * 2))
                b = 255
            else:
                # White to Red
                r = 255
                g = int(255 * (2 - norm_value * 2))
                b = int(255 * (2 - norm_value * 2))
            
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        for token, value in shapley_values.items():
            color = get_color(value)
            print(f"\033[38;2;{int(color[1:3], 16)};{int(color[3:5], 16)};{int(color[5:7], 16)}m{get_text_before_last_underscore(token)}\033[0m", end=' ')
        print()

    def _get_color(self, value, shapley_values):
        shapley_values = self.shapley_values
        norm_value = (value - min(shapley_values.values())) / (max(shapley_values.values()) - min(shapley_values.values()))
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))
    
    def plot_colored_text(self, new_line=False):
        num_items = len(self.shapley_values)
        fig_height = num_items * 0.5 + 1 if new_line else 2  # Adjust height based on new_line flag

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1  # Start from the top of the plot
        x_pos = 0.1  # Start from the left of the plot if not new_line
        step = 1 / (num_items + 1)  # Adjusted step to leave space for legend

        for sample, value in self.shapley_values.items():
            color = self._get_color(value, self.shapley_values)
            if new_line:
                ax.text(0.5, y_pos, sample.split('_')[0], color=color, fontsize=20, ha='center', va='center', transform=ax.transAxes)
                y_pos -= step  # Move down for each new word
            else:
                ax.text(x_pos, y_pos, sample.split('_')[0], color=color, fontsize=20, ha='left', va='center', transform=ax.transAxes)
                x_pos += 0.1  # Move right for each new word (adjust as needed for spacing)

        # Add a color bar as legend at the bottom
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(self.shapley_values.values()), vmax=max(self.shapley_values.values())))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])  # [left, bottom, width, height]
        cbar.set_label('Shapley Value', fontsize=12)

        plt.tight_layout()
        plt.show()

    def highlight_text_background(self):
        """
        Print each token in shapley_values with a background color 
        that varies from light yellow (near-white) to bright yellow based on the Shapley value.
        The color scale uses exponential scaling for more pronounced color differences.
        """
        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())
        
        def get_background_color(value):
            # Normalize value to 0-1 range and apply cubic scaling
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            
            # Define color transition from white to bright yellow
            r = 255  # Red remains at maximum
            g = 255  # Green remains at maximum
            b = int(255 - (norm_value * 255))  # Blue fades to zero based on scaled norm_value
            
            # Convert RGB values to ANSI background color format
            return f"\033[48;2;{r};{g};{b}m"
        
        for token, value in self.shapley_values.items():
            background_color = get_background_color(value)
            reset_color = "\033[0m"  # Reset color after each token
            print(f"{background_color}{get_text_before_last_underscore(token)}{reset_color}", end=' ')
        print()

    def analyze(self, prompt, splitter=None, sampling_ratio=0.3, print_result = False):
        if splitter is None and not hasattr(self, 'tokenizer'):
            raise ValueError("Splitter cannot be None if no tokenizer is provided.")
        self.baseline_text = self._calculate_baseline(prompt)
        token_combinations_results = self._get_result_per_token_combination(prompt, splitter, sampling_ratio)
        df_per_token_combination = self._get_df_per_token_combination(token_combinations_results, self.baseline_text)
        self.shapley_values = self._calculate_shapley_values(df_per_token_combination, prompt, splitter)
        if print_result:
            self.print_colored_text()
        
        return df_per_token_combination
