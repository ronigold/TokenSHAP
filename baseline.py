import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import random
import matplotlib.pyplot as plt
from matplotlib import colors

class NaiveBaseline:
    def __init__(self, model_name, tokenizer_path):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.corpus = []

    def random_importance(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        return {token: random.random() for token in tokens}

    def position_based_importance(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        n = len(tokens)
        return {token: (n - i) / n for i, token in enumerate(tokens)}

    def _get_color(self, value, importance_values):
        # Normalize the value between 0 and 1
        norm_value = (value - min(importance_values.values())) / (max(importance_values.values()) - min(importance_values.values()))
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))

    def plot_colored_text(self, importance_values):
        # Determine the number of items and set the figure height accordingly
        num_items = len(importance_values)
        fig_height = num_items * 0.5 + 1  # Added extra space for legend

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1  # Start from the top of the plot
        step = 1 / (num_items + 1)  # Adjusted step to leave space for legend

        for sample, value in importance_values.items():
            color = self._get_color(value, importance_values)
            ax.text(0.5, y_pos, sample, color=color, fontsize=20, ha='center', va='center', transform=ax.transAxes)
            y_pos -= step  # Move down for each new word

        # Add a color bar as legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(importance_values.values()), vmax=max(importance_values.values())))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Importance Value', fontsize=12)

        plt.tight_layout()
        plt.show()

    def analyze_and_plot(self, prompt, method='tf_idf', corpus=None):
        if method == 'random':
            importance_values = self.random_importance(prompt)
        elif method == 'position':
            importance_values = self.position_based_importance(prompt)
        else:
            raise ValueError("Invalid method. Choose 'random' or 'position'.")

        importance_values = {key.strip('Ġ'): value for key, value in importance_values.items()}\
        
        self.plot_colored_text(importance_values)
        return importance_values