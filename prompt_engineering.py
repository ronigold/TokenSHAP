import requests
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import yaml
from ollama_interact import *

class PromptEngineer:
    def __init__(self, model_name):
        self.model_name = model_name
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.api_url = config['ollama_api_url']
        
    def get_importance(self, prompt):
       
       result , _ = interact_with_ollama(model=self.model_name, prompt="""I want you to give me back JSON with the importance number for each token based on the importance of each one for the answer. Return only the JSON."

            Example 1:
            Question: "Is it raining today?"
            JSON Response: {"Is": 0.3, "it": 0.6, "raining": 0.9, "today": 0.2, "?": 0.1}

            Example 2:
            Question: "Can dogs fly?"
            JSON Response: {"Is": 0.2, "it": 0.7, "sunny": 0.1, "today": 0.3, "?": 0.8}

            Your Question:
            "Is the sky red?"
            Return the JSON.""", api_url=self.api_url, output_handler=lambda o: o)
    
       return json.loads(result)

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

    def analyze_and_plot(self, prompt):
        importance_values = self.get_importance(prompt)
        self.plot_colored_text(importance_values)