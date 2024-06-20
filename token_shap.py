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

class TokenSHAP:
    def __init__(self, model_name, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_name
        
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.ollama_api = config['ollama_api_url']
        
    def _calculate_baseline(self, prompt):
        baseline_text, _ = interact_with_ollama(model=self.model_name, prompt=prompt, api_url=  self.ollama_api, output_handler=lambda o: o)
        return baseline_text
    
    def _get_result_per_token_combination(self, prompt, splitter):
        # Check if a splitter is provided and split the prompt accordingly or tokenize if splitter is None
        if splitter is None:
            samples = self.tokenizer.tokenize(prompt)
            is_tokenized = True
        else:
            samples = prompt.split(splitter)
            is_tokenized = False

        # Dictionary to store prompts and responses
        prompt_responses = {}

        # Generate all possible subgroups of tokens
        for r in range(1, len(samples)):
            for combination in combinations(range(len(samples)), r):
                if is_tokenized:
                    # Prepare text by omitting tokens not in the current combination for tokenized input
                    omitted_tokens = [token for idx, token in enumerate(samples) if idx not in combination]
                    omitted_text = self.tokenizer.convert_tokens_to_string(omitted_tokens)
                else:
                    #Prepare text by using only the samples in the current combination for split text
                    selected_samples = [samples[idx] for idx in combination]
                    omitted_text = splitter.join(selected_samples)

                # Call the interaction function with the model
                text_response, response = interact_with_ollama(
                    model=self.model_name,
                    prompt=omitted_text,
                    api_url=self.ollama_api,
                    stream=True, 
                    output_handler=lambda o: o
                )

                # Save the prompt and response in the dictionary
                prompt_responses[omitted_text] = text_response
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
        if splitter is None:            
            tokens = self.tokenizer.tokenize(prompt)
            samples = [token.strip('Ġ') if token.startswith('Ġ') else token for token in tokens]
        else:
            samples = prompt.split('\n')
            
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
                    subset_str = " ".join(subset_samples)
                    subset_with_i_str = " ".join(subset_with_i_samples)

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
            return shapley_values
    
    def _get_color(self, value, shapley_values):
            
            # Normalize the value between 0 and 1
            norm_value = (value - min(shapley_values.values())) / (max(shapley_values.values()) - min(shapley_values.values()))
            # Create a color map that transitions from red (low) to blue (high)
            cmap = plt.cm.coolwarm
            return colors.rgb2hex(cmap(norm_value))
        
    def _plot_colored_text(self, shapley_values):
            # Determine the number of items and set the figure height accordingly
            num_items = len(shapley_values)
            fig_height = num_items * 0.5  # Adjust height factor based on preference and text size

            plt.figure(figsize=(10, fig_height))
            plt.axis('off')

            y_pos = 1  # Start from the top of the plot
            step = 1 / num_items  # Step to move each word to a new line

            for sample, value in shapley_values.items():
                plt.text(0.5, y_pos, sample, color=self._get_color(value, shapley_values), fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
                y_pos -= step  # Move down for each new word

            plt.show()
            
    def analyze(self, prompt, splitter = None):
        
        baseline_text = self. _calculate_baseline(prompt)
        token_combinations_results = self._get_result_per_token_combination(prompt, splitter)
        df_per_token_combination = self._get_df_per_token_combination(token_combinations_results, baseline_text)
        self.shapley_values = self._calculate_shapley_values(df_per_token_combination, prompt, splitter)
        self._plot_colored_text(self.shapley_values)
        
        return df_per_token_combination