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

class TokenSHAP:
    def __init__(self, model_name, tokenizer_path = None):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_name
        
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.ollama_api = config['ollama_api_url']
        
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
    
        num_total_combinations = 2**len(samples) - 1
        num_sampled_combinations = max(1, int(num_total_combinations * sampling_ratio))
    
        # Always include combinations where only one token is omitted
        essential_combinations = []
        for i in range(len(samples)):
            combination = samples[:i] + samples[i+1:]
            essential_combinations.append(combination)
    
        # Generate all possible combinations
        all_combinations = [samples]
        for r in range(1, len(samples)):
            all_combinations.extend(combinations(samples, r))
    
        # Remove essential combinations from all combinations
        remaining_combinations = [comb for comb in all_combinations if list(comb) not in essential_combinations]
    
        # Sample additional combinations
        num_additional_samples = max(0, num_sampled_combinations - len(essential_combinations))
        sampled_combinations = random.sample(remaining_combinations, 
                                             min(num_additional_samples, len(remaining_combinations)))
    
        # Combine essential and sampled combinations
        all_combinations_to_process = essential_combinations + sampled_combinations
    
        prompt_responses = {}
        for combination in all_combinations_to_process:
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
            prompt_responses[text] = text_response
        
        return prompt_responses
    
    def _get_df_per_token_combination(self, prompt_responses, baseline_text):
        df = pd.DataFrame(list(prompt_responses.items()), columns=['Prompt', 'Response'])
   
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
        shapley_values = {sample: 0 for sample in samples}
    
        # Iterate over each sample in the original sentence
        for i, sample in enumerate(samples):
            for j in range(1, n+1):
                for subset in itertools.combinations([k for k in range(n) if k != i], j-1):
                    subset = list(subset)
                    subset_with_i = subset + [i]
                    # Convert indices back to samples
                    subset_samples = [samples[k] for k in subset]
                    subset_with_i_samples = [samples[k] for k in subset_with_i]
    
                    # Find the corresponding rows in the DataFrame
                    subset_str = splitter.join(subset_samples) if splitter else " ".join(subset_samples)
                    subset_with_i_str = splitter.join(subset_with_i_samples) if splitter else " ".join(subset_with_i_samples)
    
                    v_subset = df_per_token_combination[df_per_token_combination["Prompt"].str.contains(subset_str, regex=False)]["Cosine_Similarity"].values
                    v_subset_with_i = df_per_token_combination[df_per_token_combination["Prompt"].str.contains(subset_with_i_str, regex=False)]["Cosine_Similarity"].values
    
                    # Ensure that there are matching rows
                    if len(v_subset) == 0:
                        v_subset = [0]
                    if len(v_subset_with_i) == 0:
                        v_subset_with_i = [0]
    
                    v_subset = v_subset[0]
                    v_subset_with_i = v_subset_with_i[0]
    
                    shapley_values[sample] += (factorial(len(subset)) * factorial(n - len(subset) - 1) / factorial(n)) * (v_subset_with_i - v_subset)
    
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
            print(f'\033[38;2;{int(color[1:3], 16)};{int(color[3:5], 16)};{int(color[5:7], 16)}m{token}\033[0m', end=' ')
        print()

    def _get_color(self, value, shapley_values):
        shapley_values = self.shapley_values
        norm_value = (value - min(shapley_values.values())) / (max(shapley_values.values()) - min(shapley_values.values()))
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))
    
    def plot_colored_text(self):
        # Determine the number of items and set the figure height accordingly
        num_items = len(self.shapley_values)
        fig_height = num_items * 0.5 + 1  # Added extra space for legend

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1  # Start from the top of the plot
        step = 1 / (num_items + 1)  # Adjusted step to leave space for legend

        for sample, value in self.shapley_values.items():
            color = self._get_color(value, self.shapley_values)
            ax.text(0.5, y_pos, sample, color=color, fontsize=20, ha='center', va='center', transform=ax.transAxes)
            y_pos -= step  # Move down for each new word

        # Add a color bar as legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(self.shapley_values.values()), vmax=max(self.shapley_values.values())))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Shapley Value', fontsize=12)

        plt.tight_layout()
        plt.show()

    def analyze(self, prompt, splitter=None, sampling_ratio=0.3, print_result = False):
        baseline_text = self._calculate_baseline(prompt)
        token_combinations_results = self._get_result_per_token_combination(prompt, splitter, sampling_ratio)
        df_per_token_combination = self._get_df_per_token_combination(token_combinations_results, baseline_text)
        self.shapley_values = self._calculate_shapley_values(df_per_token_combination, prompt, splitter)
        if print_result:
            self.print_colored_text()
        
        return df_per_token_combination