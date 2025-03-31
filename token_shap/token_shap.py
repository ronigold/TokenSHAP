# token_shap.py

from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.auto import tqdm
from base import BaseSHAP, TextVectorizer, ModelBase

def get_text_before_last_underscore(token: str) -> str:
    """Helper function to get text before last underscore"""
    return token.rsplit('_', 1)[0]

class Splitter:
    """Base class for text splitting"""
    def split(self, text: str) -> List[str]:
        raise NotImplementedError
        
    def join(self, tokens: List[str]) -> str:
        raise NotImplementedError

class StringSplitter(Splitter):
    """Split text by pattern (default: space)"""
    def __init__(self, split_pattern: str = ' '):
        self.split_pattern = split_pattern
    
    def split(self, prompt: str) -> List[str]:
        return re.split(self.split_pattern, prompt.strip())
    
    def join(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

class TokenizerSplitter(Splitter):
    """Split text using HuggingFace tokenizer"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split(self, prompt: str) -> List[str]:
        return self.tokenizer.tokenize(prompt)

    def join(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

class TokenSHAP(BaseSHAP):
    """Analyzes token importance in text prompts using SHAP values"""
    
    def __init__(self, 
                 model: ModelBase,
                 splitter: Splitter,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        """
        Initialize TokenSHAP
        
        Args:
            model: Model to analyze
            splitter: Text splitter implementation
            vectorizer: Text vectorizer for calculating similarities
            debug: Enable debug output
        """
        super().__init__(model=model, vectorizer=vectorizer, debug=debug)
        self.splitter = splitter

    def _prepare_generate_args(self, content: str, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        return {"prompt": content}

    def _get_samples(self, content: str) -> List[str]:
        """Get tokens from prompt"""
        return self.splitter.split(content)

    def _prepare_combination_args(self, combination: List[str], original_content: str) -> Dict:
        """Prepare model arguments for a combination"""
        return {"prompt": self.splitter.join(combination)}

    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        text = self.splitter.join(combination)
        return text + '_' + ','.join(str(index) for index in indexes)

    def print_colored_text(self):
        """Print text with tokens colored by importance"""
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before visualization")

        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

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

        for token, value in self.shapley_values.items():
            color = get_color(value)
            print(
                f"\033[38;2;{int(color[1:3], 16)};"
                f"{int(color[3:5], 16)};"
                f"{int(color[5:7], 16)}m"
                f"{get_text_before_last_underscore(token)}\033[0m",
                end=' '
            )
        print()

    def plot_colored_text(self, new_line: bool = False):
        """
        Plot text visualization with importance colors
        
        Args:
            new_line: Whether to plot tokens on new lines
        """
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before visualization")

        num_items = len(self.shapley_values)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)

        for sample, value in self.shapley_values.items():
            norm_value = (value - min(self.shapley_values.values())) / (
                max(self.shapley_values.values()) - min(self.shapley_values.values())
            )
            color = plt.cm.coolwarm(norm_value)

            if new_line:
                ax.text(
                    0.5, y_pos, 
                    get_text_before_last_underscore(sample), 
                    color=color, 
                    fontsize=20,
                    ha='center', 
                    va='center', 
                    transform=ax.transAxes
                )
                y_pos -= step
            else:
                ax.text(
                    x_pos, y_pos, 
                    get_text_before_last_underscore(sample), 
                    color=color, 
                    fontsize=20,
                    ha='left', 
                    va='center', 
                    transform=ax.transAxes
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
        """Print text with background colors based on importance"""
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before visualization")

        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

        for token, value in self.shapley_values.items():
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            background_color = f"\033[48;2;{r};{g};{b}m"
            reset_color = "\033[0m"
            print(
                f"{background_color}"
                f"{get_text_before_last_underscore(token)}"
                f"{reset_color}",
                end=' '
            )
        print()

    def analyze(self, prompt: str, 
                sampling_ratio: float = 0.0,
                max_combinations: Optional[int] = 1000,
                print_highlight_text: bool = False) -> pd.DataFrame:
        """
        Analyze token importance in a prompt
        
        Args:
            prompt: Text prompt to analyze
            sampling_ratio: Ratio of combinations to sample (0-1)
            max_combinations: Maximum number of combinations to generate
            print_highlight_text: Whether to print highlighted text after analysis
            
        Returns:
            DataFrame with analysis results
        """
        # Clean prompt
        prompt = prompt.strip()
        prompt = re.sub(r'\s+', ' ', prompt)

        # Get baseline and process combinations using base class methods
        self.baseline_text = self._calculate_baseline(prompt)
        responses = self._get_result_per_combination(
            prompt, 
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations  # Pass max_combinations to base class
        )
        
        # Create results DataFrame
        self.results_df = self._get_df_per_combination(responses, self.baseline_text)
        
        # Calculate Shapley values
        self.shapley_values = self._calculate_shapley_values(self.results_df, prompt)

        if print_highlight_text:
            self.highlight_text_background()

        return self.results_df