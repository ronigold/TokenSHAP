#pixel_shap.py

from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import os
from tqdm.auto import tqdm
from .base import BaseSHAP, TextVectorizer, ModelBase
from .image_utils import BaseSegmentationModel, SegmentationBased
from .visualization import PixelSHAPVisualizer

class PixelSHAP(BaseSHAP):
    """Analyzes object importance in images using segmentation masks"""
    
    def __init__(
            self,
            model: ModelBase,
            segmentation_model: BaseSegmentationModel,
            manipulator: SegmentationBased,
            vectorizer: Optional[TextVectorizer] = None,
            debug: bool = False,
            temp_dir: str = 'temp_images'
        ):
        super().__init__(model=model, vectorizer=vectorizer, debug=debug)
        self.segmentation_model = segmentation_model
        self.manipulator = manipulator
        self.temp_dir = temp_dir
        self.visualizer = PixelSHAPVisualizer()
        os.makedirs(self.temp_dir, exist_ok=True)
        self._current_labels = None
        self._current_masks = None

    def _detect_objects(self, image_path: Union[str, Path], return_segmentation: bool = False) -> Any:
        """Helper to detect objects and store results"""
        boxes, labels, scores, masks = self.segmentation_model.segment(image_path)
        if boxes is None or masks is None or len(boxes) == 0:
            raise ValueError(f"No objects detected in image: {image_path}")
        self._current_labels = labels
        self._current_masks = masks
        
        if self.debug:
            print(f"\nDetected objects:")
            for i, label in enumerate(labels):
                print(f"  {i}: {label}")
        if return_segmentation:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return boxes, labels, scores, masks, image

    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        if isinstance(content, dict):
            return {
                "prompt": content.get("prompt", ""),
                "image_path": content.get("image_path", "")
            }
        return {
            "prompt": kwargs.get("prompt", ""),
            "image_path": str(content)
        }
    
    def _get_samples(self, content: Any) -> List[str]:
        """
        Get list of object labels (with indices) instead of just indices.
        Example return: ["person_0", "sports ball_1", "person_2"] 
        """
        image_path = content.get("image_path", content)
        self._detect_objects(image_path)
        
        return [f"{label}_{i}" for i, label in enumerate(self._current_labels)]
    
    def _prepare_combination_args(self, combination: List[str], original_content: Any) -> Dict:
        """
        הכוונה: combination = רשימת האובייקטים שאנו רוצים להשאיר גלויים.
        לכן נמחק את כל מה שלא ב-combination.
        """
        image_path = original_content.get("image_path", original_content)
        prompt = original_content.get("prompt", "")
    
        # Load the original image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        modified = image.copy()
    
        all_objs = [f"{label}_{i}" for i, label in enumerate(self._current_labels)]
    
        objs_to_hide = [obj for obj in all_objs if obj not in combination]
        
        # Get the indices of objects to preserve (those in the combination)
        preserve_indices = []
        for obj_str in combination:
            idx = int(obj_str.split('_')[-1])
            preserve_indices.append(idx)
    
        # Hide each object that's not in the combination
        for obj_str in objs_to_hide:
            idx = int(obj_str.split('_')[-1])
            # Pass preserve_indices to manipulator to avoid modifying pixels of objects to keep
            modified = self.manipulator.manipulate(
                modified, 
                self._current_masks, 
                idx,
                preserve_indices=preserve_indices
            )
    
        temp_name = f"temp_{','.join([obj.split('_')[-1] for obj in combination])}.jpg"
        temp_path = os.path.join(self.temp_dir, temp_name)
        cv2.imwrite(temp_path, cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
    
        if self.debug:
            print(f"\nPreparing combination:")
            print(f"  Showing: {', '.join(combination)}")
            print(f"  Hiding: {', '.join(objs_to_hide)}")
            print(f"  Preserved indices: {preserve_indices}")
    
        return {"prompt": prompt, "image_path": temp_path}


    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        # Simply return a descriptive key (we also store combination fully in 'responses')
        return f"combination_{','.join(map(str, indexes))}"

    def _get_result_per_combination(self, 
                                    content: Any, 
                                    sampling_ratio: float,
                                    max_combinations: Optional[int] = None
                                   ) -> Dict[str, Tuple[str, Tuple[int, ...], List[str]]]:
        """
        Slight override to store 'combination' as well in the response.
        Everything else stays as in BaseSHAP, except at the end we do:
            responses[key] = (response, indexes, combination)
        """
        # -- COPY of base logic, except for storing combination in responses --
        samples = self._get_samples(content)
        n = len(samples)
        self._debug_print(f"Number of samples: {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        # Always start with essential combinations (each missing one sample)
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combo = samples[:i] + samples[i + 1:]
            idxs = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combo, idxs))
            essential_combinations_set.add(idxs)
        
        num_essential = len(essential_combinations)
        self._debug_print(f"Number of essential combinations: {num_essential}")
        if max_combinations is not None and max_combinations < num_essential:
            print(f"Warning: max_combinations ({max_combinations}) is less than the number of essential combinations "
                  f"({num_essential}). Will use all essential combinations despite the limit.")
            self._debug_print("No additional combinations will be added.")
            max_combinations = num_essential
        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)
            self._debug_print(f"Remaining combinations budget after essentials: {remaining_budget}")

        # If using sampling ratio, calculate possible additional combos
        if sampling_ratio < 1.0:
            theoretical_total = 2 ** n - 1
            theoretical_additional = theoretical_total - num_essential
            desired_additional = int(theoretical_additional * sampling_ratio)
            num_additional = min(desired_additional, remaining_budget)
        else:
            num_additional = remaining_budget

        num_additional = int(num_additional)
        self._debug_print(f"Number of additional combinations to sample: {num_additional}")

        # Additional random combos
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(additional_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        all_combinations = essential_combinations + additional_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations)}")

        responses = {}
        for idx, (combo, idxs) in enumerate(tqdm(all_combinations, desc="Processing combinations")):
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations)}:")
            self._debug_print(f"Combination: {combo}")
            self._debug_print(f"Indexes: {idxs}")

            args = self._prepare_combination_args(combo, content)
            response = self.model.generate(**args)
            self._debug_print(f"Received response for combination {idx + 1}")

            key = self._get_combination_key(combo, idxs)
            # ⬇ Store (response, indexes, combo) instead of just (response, indexes)
            responses[key] = (response, idxs, list(combo))

        return responses

    def _get_df_per_combination(self, 
                                responses: Dict[str, Tuple[str, Tuple[int, ...], List[str]]], 
                                baseline_text: str
                               ) -> pd.DataFrame:
        """
        Create a DataFrame with combination results, but now we rely on the stored 'combo'
        to show exactly which objects were shown/hidden. 
        """
        rows = []
        # Full list of objects from _current_labels:
        all_objs = [f"{lbl}_{i}" for i, lbl in enumerate(self._current_labels)]

        for key, (response, indexes, combination) in responses.items():
            # 'combination' is the actual list of strings used, e.g. ["sports ball_1", "person_2"]
            shown = combination
            hidden = [obj for obj in all_objs if obj not in shown]

            content_desc = f"Showing: {', '.join(shown)} | Hidden: {', '.join(hidden)}"

            rows.append({
                'Combination_Key': key,        # e.g. "combination_1,2" 
                'Used_Combination': shown,     # the actual objects shown
                'Hidden_Objects': hidden,
                'Response': response,
                'Indexes': indexes            # still keep the 1-based indexes for reference
            })
        
        df = pd.DataFrame(rows)

        # Calculate similarities
        all_texts = [baseline_text] + df["Response"].tolist()
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["Similarity"] = similarities
        
        if self.debug:
            print("\nDataFrame summary:")
            for _, row in df.iterrows():
                print(f"\nCombination_Key: {row['Combination_Key']}")
                print(f"  Content: {row['Used_Combination']} | Hidden: {row['Hidden_Objects']}")
                print(f"  Similarity: {row['Similarity']:.3f}")
        
        return df

    def analyze(self, 
                image_path: Union[str, Path], 
                prompt: str,
                sampling_ratio: float = 0.5,
                max_combinations: Optional[int] = None,
                cleanup_temp_files: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Analyze object importance in an image, then fix Shapley keys so they don't have 
        the trailing '_1','_2', etc.
        """

        if self.debug:
            print(f"\nStarting analysis:")
            print(f"Image: {image_path}")
            print(f"Prompt: {prompt}")
        self._last_image_path = str(image_path)
        self._last_prompt = prompt
        
        content = {"image_path": str(image_path), "prompt": prompt}
        
        # 1. Baseline
        self.baseline_text = self._calculate_baseline(content)
        
        # 2. Process combinations
        #    (calls our custom _get_result_per_combination that also stores the combination)
        responses = self._get_result_per_combination(
            content, 
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations
        )
    
        # 3. Create results DataFrame
        self.results_df = self._get_df_per_combination(responses, self.baseline_text)
        
        # 4. Calculate Shapley values 
        raw_shapley_values = self._calculate_shapley_values(self.results_df, content)
        
        # 5. Fix Shapley keys so we don't have "person_0_1", etc.
        fixed_shapley_values = {}
        for key, val in raw_shapley_values.items():
            parts = key.rsplit('_', 1)  
            if len(parts) == 2 and parts[1].isdigit():
                corrected_key = parts[0]
            else:
                corrected_key = key       
            fixed_shapley_values[corrected_key] = val
    
        self.shapley_values = fixed_shapley_values
        
        if self.debug:
            print("\nCorrected Shapley Keys:")
            for k,v in self.shapley_values.items():
                print(f"  {k}: {v}")
        
        # 6. Cleanup if requested
        if cleanup_temp_files:
            for file in os.listdir(self.temp_dir):
                if file.startswith('temp_'):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except Exception as e:
                        if self.debug:
                            print(f"Warning: Could not remove {file}: {e}")
    
        return self.results_df, self.shapley_values
        
    def visualize(self,
                     show_labels: bool = False,
                     heatmap_style: bool = True,
                     output_path: Optional[str] = None,
                     overlay_original: bool = True,
                     show_colorbar=True,
                     show_original_side_by_side=False,
                     original_opacity: float = 0.2,
                     heatmap_opacity: float = 0.7,
                     background_opacity: float = 0.2,
                     thickness=1,
                     roughness=2,
                     show_legend=False,
                     show_model_output=False,
                     **kwargs):
        """
        Create visualization of object importance
        
        Args:
            show_labels: Whether to show object labels
            heatmap_style: Whether to use heatmap coloring
            output_path: Optional path to save visualization
            overlay_original: Whether to blend visualization with original image
            original_opacity: Opacity of original image when overlaying (0-1)
            heatmap_opacity: Opacity of heatmap colors (0-1)
            background_opacity : float, optional (default=0.0)
            **kwargs: Additional visualization parameters
        """
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before visualization")
            
        if not hasattr(self, '_last_image_path'):
            raise ValueError("No image path found. Make sure analyze() was called successfully")
        
        boxes, labels, scores, masks, image = self._detect_objects(self._last_image_path, return_segmentation=True)
        
        self.visualizer.plot_importance_transparent(
            image_path=self._last_image_path,
            masks=masks,
            shapley_values=self.shapley_values,
            labels=labels,
            output_path=output_path,
            show_labels=show_labels,
            heatmap_style=heatmap_style,
            overlay_original=overlay_original,
            original_opacity=original_opacity,
            heatmap_opacity=heatmap_opacity,
            show_colorbar=show_colorbar,
            background_opacity=background_opacity,
            prompt=self._last_prompt,
            show_original_side_by_side=show_original_side_by_side,
            thickness=thickness,
            roughness=roughness,
            show_legend=show_legend,
            model_output=self.baseline_text if show_model_output else None,
            **kwargs
        )

    def plot_importance_ranking(
        self,
        thumbnail_size=60,
        show_values=True,
    ):
        boxes, labels, scores, masks, image = self._detect_objects(self._last_image_path, return_segmentation=True)
        image = cv2.imread(str(self._last_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.visualizer.plot_importance_ranking(
            shapley_values=self.shapley_values,  
            image=image,
            masks=masks,
            thumbnail_size=60,
            show_values=True
        )

    def extract_masks_with_values(self) -> dict:
        """
        Extract the segmentation masks with their corresponding Shapley values.
        Objects are sorted by Shapley value (highest first).
        
        Returns:
            dict: A dictionary containing:
                - 'image': The original image (RGB)
                - 'masks_with_values': List of dictionaries containing masks, bounding boxes, labels and Shapley values
                                       (sorted by shapley_value in descending order)
                - 'boxes': List of bounding boxes from the segmentation model
                - 'labels': List of object labels
                - 'shapley_values': Dictionary mapping labels to their Shapley values
                - 'normalized_values': Dictionary mapping labels to normalized Shapley values (0-1 range)
        """
        if not hasattr(self, 'shapley_values'):
            raise ValueError("Must run analyze() before extracting masks with values")
                
        if not hasattr(self, '_last_image_path'):
            raise ValueError("No image path found. Make sure analyze() was called successfully")
        
        # Re-detect objects to get masks
        boxes, labels, scores, masks, image = self._detect_objects(self._last_image_path, return_segmentation=True)
        
        if self.debug:
            print(f"Detected {len(masks)} objects")
            print(f"Detected labels: {labels}")
            print(f"Shapley values: {self.shapley_values}")
            
        # Print a warning if we have empty Shapley values
        if not self.shapley_values:
            print("WARNING: Shapley values dictionary is empty! Analysis may not have completed properly.")
        
        # Print a warning if all Shapley values are zero
        if all(val == 0 for val in self.shapley_values.values()):
            print("WARNING: All Shapley values are zero! Analysis may not have produced meaningful results.")
        
        # Normalize Shapley values to 0-1 range for easier visualization if needed
        values = list(self.shapley_values.values())
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        range_val = max_val - min_val
        
        normalized_values = {}
        if range_val > 0:  # Avoid division by zero
            for key, val in self.shapley_values.items():
                normalized_values[key] = (val - min_val) / range_val
        else:
            # If all values are the same, set them to 0.5
            for key in self.shapley_values:
                normalized_values[key] = 0.5
        
        # Create a list of mask dictionaries with their values
        masks_with_values = []
        for i, mask in enumerate(masks):
            label = labels[i]
            box = boxes[i] if i < len(boxes) else None
            # Get the key without index suffix (e.g., "person" not "person_0")
            # For debugging, also show the original label with index
            label_with_idx = f"{label}_{i}"
            
            # Find the Shapley value for this object
            shapley_value = None
            
            # Exact match
            if label in self.shapley_values:
                shapley_value = self.shapley_values[label]
            
            # Try with index suffix
            if shapley_value is None:
                key_with_index = f"{label}_{i}"
                if key_with_index in self.shapley_values:
                    shapley_value = self.shapley_values[key_with_index]
                    if self.debug:
                        print(f"Found Shapley value using indexed key {key_with_index}: {shapley_value}")
            
            # If still not found, try fuzzy matching
            if shapley_value is None:
                for key, val in self.shapley_values.items():
                    # Match by checking if key starts with the label (ignoring index)
                    if key.startswith(f"{label}_") or key == label:
                        shapley_value = val
                        if self.debug:
                            print(f"Found Shapley value using fuzzy match {key}: {shapley_value}")
                        break
            
            # If still not found, default to 0
            if shapley_value is None:
                if self.debug:
                    print(f"WARNING: No Shapley value found for {label}")
                shapley_value = 0.0
            
            # Get the normalized value
            normalized_value = normalized_values.get(label, 0.0)
            
            masks_with_values.append({
                'mask': mask,
                'box': box,
                'label': label,
                'index': i,
                'label_with_idx': label_with_idx,
                'shapley_value': shapley_value,
                'normalized_value': normalized_value
            })
        
        # Sort by importance (highest Shapley value first)
        masks_with_values.sort(key=lambda x: x['shapley_value'], reverse=True)
        
        if self.debug:
            print("\nObjects sorted by Shapley value (highest first):")
            for obj in masks_with_values:
                print(f"  {obj['label']}: {obj['shapley_value']}")
        
        return {
            'image': image,
            'masks_with_values': masks_with_values,  # Already sorted by Shapley value (highest first)
            'boxes': boxes,
            'labels': labels,
            'shapley_values': self.shapley_values,
            'normalized_values': normalized_values
        }