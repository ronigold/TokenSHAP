#image_utils.py

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import cv2
from abc import ABC, abstractmethod
import torch
from torchvision.ops import box_convert
import torchvision.transforms as transforms
import os
import sys

# Import Transformers SAM2 - no local file dependencies needed!
from transformers import Sam2Model, Sam2Processor

class BaseObjectDetector:
    """Base class for object detection models"""
    
    def detect(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes in xyxy format, class labels, confidence scores)
        """
        raise NotImplementedError()

# Optional imports for additional functionality
Sam2VideoTracker = None
DinoDetectionModel = None

# Try to import YOLO if available
try:
    from ultralytics import YOLO as UltralyticsYOLO
    
    class YOLODetectionModel(BaseObjectDetector):
        """YOLO detection model using Ultralytics"""
        
        def __init__(self, model_path: str, target_categories: List[str] = None, 
                     device: str = "cuda", conf_threshold: float = 0.25, 
                     iou_threshold: float = 0.45, use_yolov8: bool = True):
            self.model = UltralyticsYOLO(model_path)
            self.device = device
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            self.target_categories = target_categories if target_categories else []
            
        def detect(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
            """Detect objects in an image using YOLO"""
            results = self.model(str(image_path), conf=self.conf_threshold, 
                                iou=self.iou_threshold, device=self.device)
            
            if len(results) == 0 or results[0].boxes is None:
                return None, None, None
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            labels = [results[0].names[class_id] for class_id in class_ids]
            
            # Filter by target categories if specified
            if self.target_categories:
                filtered_indices = [i for i, label in enumerate(labels) 
                                  if label in self.target_categories]
                if filtered_indices:
                    boxes = boxes[filtered_indices]
                    labels = [labels[i] for i in filtered_indices]
                    confidences = confidences[filtered_indices]
                else:
                    return None, None, None
            
            return boxes, labels, confidences.tolist()
            
except ImportError:
    YOLODetectionModel = None

# Only try to import DynamicSAM2 if explicitly configured (for backward compatibility)
if "DYNAMIC_SAM2_PATH" in os.environ:
    try:
        sys.path.append(os.environ["DYNAMIC_SAM2_PATH"])
        from dynamic_sam2.sam2_video_tracker import Sam2VideoTracker
        from dynamic_sam2.object_detection import YOLODetectionModel, DinoDetectionModel
    except ImportError as e:
        print(f"Note: DynamicSAM2 configured but import failed: {e}")

# Try to import Transformers Grounding DINO if available
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image as PILImage
    
    class TransformersDinoDetector(BaseObjectDetector):
        """Grounding DINO detector using HuggingFace Transformers"""
        
        def __init__(self, 
                     text_prompt: str,
                     model_id: str = "IDEA-Research/grounding-dino-tiny",
                     device: str = "cuda",
                     box_threshold: float = 0.35,
                     text_threshold: float = 0.25):
            self.text_prompt = text_prompt
            self.model_id = model_id
            self.device = device if torch.cuda.is_available() else "cpu"
            self.box_threshold = box_threshold
            self.text_threshold = text_threshold
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
        def detect(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
            """Detect objects based on text prompt"""
            # Load image as PIL Image
            image = PILImage.open(str(image_path))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Convert PIL image to numpy array for the processor
            image_np = np.array(image)
            
            # Parse text labels - ensure they end with period for better detection
            labels = [label.strip() for label in self.text_prompt.split(",")]
            # Join labels with periods for better detection
            text_input = ". ".join(labels)
            if not text_input.endswith("."):
                text_input = text_input + "."
            
            # Process inputs - text should be a string, not a list
            inputs = self.processor(
                images=image_np,  # Use numpy array
                text=text_input,  # Pass as string
                return_tensors="pt"
            ).to(self.device)
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]
            )
            
            if len(results) == 0 or len(results[0]["boxes"]) == 0:
                return None, None, None
                
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy().tolist()
            
            # Get labels - handle different key names and formats
            if "labels" in result:
                # Labels might be indices or text
                raw_labels = result["labels"]
                if isinstance(raw_labels[0], str):
                    labels = raw_labels
                else:
                    # If they're indices, map them back to our text labels
                    original_labels = [label.strip() for label in self.text_prompt.split(",")]
                    labels = []
                    for idx in raw_labels:
                        if isinstance(idx, torch.Tensor):
                            idx = idx.item()
                        # Map index to label (may repeat if multiple instances)
                        if idx < len(original_labels):
                            labels.append(original_labels[idx])
                        else:
                            labels.append(f"object_{idx}")
            else:
                # Fallback to empty labels
                labels = [f"object_{i}" for i in range(len(boxes))]
            
            # Clean up labels - remove any trailing periods or extra whitespace
            labels = [label.strip().rstrip(".") for label in labels]
            
            return boxes, labels, scores
    
    TRANSFORMERS_DINO_AVAILABLE = True
except ImportError:
    TransformersDinoDetector = None
    TRANSFORMERS_DINO_AVAILABLE = False

class BaseSegmentationModel:
    """Base class for object segmentation models"""
    
    def segment(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]], Optional[np.ndarray]]:
        """
        Detect and segment objects in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes in xyxy format, class labels, confidence scores, segmentation masks)
            where masks is a numpy array of shape (N, H, W) with N binary masks
        """
        raise NotImplementedError()

class Sam2Adapter:
    """Adapter for SAM2 segmentation model using Transformers"""
    
    def __init__(
        self,
        sam2_model_id: str = "facebook/sam2.1-hiera-large",
        sam2_cfg_path: str = None,  # Deprecated - kept for backward compatibility
        sam2_ckpt_path: str = None,  # Deprecated - kept for backward compatibility
        device: str = "cuda",
        mask_threshold: float = 0.5,  # Higher threshold for cleaner boundaries
        use_stability_score: bool = True,  # Consider stability in mask selection
        stability_score_thresh: float = 0.95,  # Minimum stability score
        remove_small_regions: bool = True,  # Post-process to remove small regions
        min_region_area: int = 100,  # Minimum area for regions to keep
        resolve_overlaps: bool = True,  # Resolve overlapping masks
        overlap_resolution: str = "score",  # How to resolve: "score", "bbox_center", "first"
        mask_selection: str = "largest",  # How to select from 3 masks: "iou", "largest", "coverage"
        clip_to_bbox: bool = True  # Whether to clip masks to bounding boxes
    ):
        """
        Initialize SAM2 adapter using Transformers
        
        Args:
            sam2_model_id: HuggingFace model ID (e.g., "facebook/sam2.1-hiera-large")
            sam2_cfg_path: (Deprecated) Legacy parameter, ignored
            sam2_ckpt_path: (Deprecated) Legacy parameter, ignored
            device: Device to run model on
            mask_threshold: Threshold for mask binarization (higher = cleaner boundaries)
            use_stability_score: Whether to consider stability when selecting masks
            stability_score_thresh: Minimum stability score for mask selection
            remove_small_regions: Whether to remove small disconnected regions
            min_region_area: Minimum area for regions to keep
            resolve_overlaps: Whether to resolve overlapping regions between masks
            overlap_resolution: Method to resolve overlaps:
                - "score": Assign to object with higher IoU score
                - "bbox_center": Assign to object whose bbox center is closer
                - "first": Keep first object's mask (by order)
        """
        self.device = device
        self.mask_threshold = mask_threshold
        self.use_stability_score = use_stability_score
        self.stability_score_thresh = stability_score_thresh
        self.remove_small_regions = remove_small_regions
        self.min_region_area = min_region_area
        self.resolve_overlaps = resolve_overlaps
        self.overlap_resolution = overlap_resolution
        self.mask_selection = mask_selection
        self.clip_to_bbox = clip_to_bbox
        self.iou_scores = None  # Store IoU scores for overlap resolution
        
        # Always use Transformers SAM2
        if sam2_cfg_path or sam2_ckpt_path:
            print("Note: sam2_cfg_path and sam2_ckpt_path are deprecated. Using HuggingFace model instead.")
            
        # Use provided model ID or default
        self.sam2_model_id = sam2_model_id if sam2_model_id else "facebook/sam2.1-hiera-large"
        
        self._initialize_sam2()
        
    def _initialize_sam2(self):
        """Initialize SAM2 model from Transformers"""
        try:
            import warnings
            import logging
            
            # Suppress the specific warning about model type mismatch
            # This is expected since all SAM2 models on HuggingFace are video models
            # but they work fine for image segmentation
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Initialize model and processor from HuggingFace
                self.model = Sam2Model.from_pretrained(self.sam2_model_id).to(self.device)
                self.processor = Sam2Processor.from_pretrained(self.sam2_model_id)
            
            # Restore logging level
            logging.getLogger("transformers").setLevel(logging.WARNING)
            
            self.model.eval()
        except ImportError as e:
            raise ImportError(f"Transformers library not installed or SAM2 not available: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model '{self.sam2_model_id}': {str(e)}")
    
    def generate_masks_from_boxes(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Generate precise masks from bounding boxes using SAM2
        
        Args:
            image: RGB image as numpy array
            boxes: Array of bounding boxes in xyxy format
            
        Returns:
            Array of binary masks with shape (N, H, W)
        """
        if len(boxes) == 0:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            
        # Ensure RGB format
        if image.shape[2] == 3 and image[0,0,0] > image[0,0,2]:  # Simple BGR check
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        import torch
        from PIL import Image
        
        # Convert numpy array to PIL Image for processor
        pil_image = Image.fromarray(image_rgb)
        
        # Process all boxes at once for consistency
        # SAM2 expects boxes as nested list: [list of boxes for single image]
        input_boxes = [boxes.tolist()]  # Single image with multiple boxes
        
        # Process inputs
        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate masks with multimask_output=True to get multiple options
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)
        
        # Get the predictions
        pred_masks = outputs.pred_masks.cpu()  # Shape: [batch, num_objects, num_masks, H, W]
        iou_scores = outputs.iou_scores.cpu()  # Shape: [batch, num_objects, num_masks]
        
        # Select best mask for each object with stability consideration
        batch_idx = 0  # We're processing single image
        num_objects = pred_masks.shape[1]
        
        # Create tensor to hold best masks and scores
        best_masks = []
        best_iou_scores = []
        
        for obj_idx in range(num_objects):
            obj_masks = pred_masks[batch_idx, obj_idx]  # Shape: [num_masks, H, W]
            obj_scores = iou_scores[batch_idx, obj_idx]  # Shape: [num_masks]
            
            # Select best mask based on strategy
            if self.mask_selection == "largest":
                # Choose mask with most pixels (best coverage)
                mask_areas = []
                for mask_idx in range(obj_masks.shape[0]):
                    # Apply threshold to get binary mask
                    binary_mask = obj_masks[mask_idx] > self.mask_threshold
                    area = torch.sum(binary_mask).item()
                    mask_areas.append(area)
                best_idx = torch.argmax(torch.tensor(mask_areas))
                
            elif self.mask_selection == "coverage":
                # Choose mask with best coverage within bounding box
                # Need the bounding box for this
                if obj_idx < len(boxes):
                    x1, y1, x2, y2 = boxes[obj_idx].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(obj_masks.shape[2], x2), min(obj_masks.shape[1], y2)
                    
                    coverages = []
                    for mask_idx in range(obj_masks.shape[0]):
                        mask = obj_masks[mask_idx]
                        # Calculate coverage in bbox region
                        bbox_mask = mask[y1:y2, x1:x2]
                        coverage = torch.sum(bbox_mask > self.mask_threshold).item()
                        coverages.append(coverage)
                    best_idx = torch.argmax(torch.tensor(coverages))
                else:
                    # Fallback to largest if no bbox available
                    mask_areas = []
                    for mask_idx in range(obj_masks.shape[0]):
                        binary_mask = obj_masks[mask_idx] > self.mask_threshold
                        area = torch.sum(binary_mask).item()
                        mask_areas.append(area)
                    best_idx = torch.argmax(torch.tensor(mask_areas))
                    
            else:  # "iou" or default
                # Original behavior - use IoU scores
                if self.use_stability_score:
                    # Compute stability scores for each mask
                    stability_scores = []
                    for mask_idx in range(obj_masks.shape[0]):
                        mask = obj_masks[mask_idx]
                        stability = self._compute_stability_score(mask)
                        stability_scores.append(stability)
                    
                    stability_scores = torch.tensor(stability_scores, device=obj_scores.device)
                    
                    # Filter masks by stability threshold
                    stable_mask = stability_scores >= self.stability_score_thresh
                    
                    if stable_mask.any():
                        # Among stable masks, choose the one with highest IoU
                        obj_scores_filtered = obj_scores.clone()
                        obj_scores_filtered[~stable_mask] = -1  # Set unstable masks to low score
                        best_idx = torch.argmax(obj_scores_filtered)
                    else:
                        # If no stable masks, fall back to highest IoU
                        best_idx = torch.argmax(obj_scores)
                else:
                    # Simply use highest IoU score
                    best_idx = torch.argmax(obj_scores)
                
            best_masks.append(obj_masks[best_idx])
            best_iou_scores.append(obj_scores[best_idx].item())
        
        if best_masks:
            # Stack best masks into single tensor
            best_masks_tensor = torch.stack(best_masks, dim=0)  # Shape: [num_objects, H, W]
            
            # Add batch and channel dimensions for post-processing
            # post_process_masks expects shape [batch, num_channels, H, W]
            masks_for_processing = best_masks_tensor.unsqueeze(0).unsqueeze(2)  # Shape: [1, num_objects, 1, H, W]
            
            # Post-process all masks at once with our threshold
            processed_masks = self.processor.post_process_masks(
                masks_for_processing,
                inputs["original_sizes"],
                mask_threshold=self.mask_threshold,
                binarize=True
            )
            
            # Extract masks from batch output
            if isinstance(processed_masks, list):
                masks_np = processed_masks[0]  # Get first (and only) image
            else:
                masks_np = processed_masks
                
            # Convert to numpy if needed
            if isinstance(masks_np, torch.Tensor):
                masks_np = masks_np.numpy()
            
            # Remove the extra channel dimension if present
            # Shape should be [num_objects, 1, H, W] -> [num_objects, H, W]
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np.squeeze(1)
            
            # Ensure each mask is within its bounding box
            all_masks = []
            for obj_idx in range(min(len(boxes), masks_np.shape[0])):
                mask = masks_np[obj_idx]
                
                if self.clip_to_bbox:
                    # Clip mask to bounding box
                    x1, y1, x2, y2 = boxes[obj_idx].astype(int)
                    # Ensure bounds are within image
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
                    
                    # Create a mask that's only True within the bounding box
                    clipped_mask = np.zeros_like(mask, dtype=bool)
                    clipped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                else:
                    # Use full mask without clipping
                    clipped_mask = mask
                
                # Remove small disconnected regions if requested
                if self.remove_small_regions:
                    clipped_mask = self._remove_small_regions(clipped_mask, self.min_region_area)
                
                all_masks.append(clipped_mask)
        else:
            all_masks = []
        
        # Stack all masks
        if all_masks:
            masks_array = np.stack(all_masks, axis=0)
            
            # Resolve overlaps if requested
            if self.resolve_overlaps and len(masks_array) > 1:
                # Use the IoU scores we collected earlier
                masks_array = self._resolve_overlaps(
                    masks_array, 
                    boxes, 
                    best_iou_scores if len(best_iou_scores) > 0 else None
                )
            
            return masks_array.astype(bool)
        else:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
    
    def _compute_stability_score(self, mask: torch.Tensor, threshold_delta: float = 0.05) -> float:
        """
        Compute stability score for a mask.
        A stable mask changes little when the threshold is varied.
        
        Args:
            mask: Mask tensor
            threshold_delta: Delta for threshold variation
            
        Returns:
            Stability score between 0 and 1
        """
        # Compute areas at different thresholds
        area_original = (mask > 0.0).sum().item()
        area_high = (mask > threshold_delta).sum().item()
        area_low = (mask > -threshold_delta).sum().item()
        
        if area_low == 0:
            return 0.0
        
        # Stability is ratio of high threshold area to low threshold area
        stability = area_high / area_low
        return min(stability, 1.0)
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove small disconnected regions from a binary mask.
        
        Args:
            mask: Binary mask
            min_area: Minimum area for regions to keep
            
        Returns:
            Cleaned mask with small regions removed
        """
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return mask
        
        # Calculate area of each component
        areas = ndimage.sum(mask, labeled, range(1, num_features + 1))
        
        # Keep only components larger than min_area
        cleaned_mask = np.zeros_like(mask)
        for i, area in enumerate(areas, 1):
            if area >= min_area:
                cleaned_mask[labeled == i] = True
        
        return cleaned_mask
    
    def _resolve_overlaps(self, masks: np.ndarray, boxes: np.ndarray, iou_scores: Optional[List[float]] = None) -> np.ndarray:
        """
        Resolve overlapping regions between masks.
        
        Args:
            masks: Array of binary masks with shape (N, H, W)
            boxes: Array of bounding boxes with shape (N, 4)
            iou_scores: Optional list of IoU scores for each mask
            
        Returns:
            Masks with overlaps resolved
        """
        n_masks = masks.shape[0]
        h, w = masks.shape[1:]
        
        # Create a priority map to track which mask owns each pixel
        priority_map = np.zeros((h, w), dtype=np.int32) - 1  # -1 means no owner
        
        if self.overlap_resolution == "score" and iou_scores is not None:
            # Sort masks by IoU score (highest first)
            priorities = np.argsort(iou_scores)[::-1]
        elif self.overlap_resolution == "bbox_center":
            # Sort by distance from bbox center to mask centroid
            priorities = []
            for i in range(n_masks):
                if masks[i].any():
                    # Calculate mask centroid
                    y_coords, x_coords = np.where(masks[i])
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    
                    # Calculate bbox center
                    x1, y1, x2, y2 = boxes[i]
                    bbox_center_x = (x1 + x2) / 2
                    bbox_center_y = (y1 + y2) / 2
                    
                    # Calculate distance
                    dist = np.sqrt((centroid_x - bbox_center_x)**2 + (centroid_y - bbox_center_y)**2)
                    priorities.append((i, dist))
                else:
                    priorities.append((i, float('inf')))
            
            # Sort by distance (smallest first)
            priorities = [idx for idx, _ in sorted(priorities, key=lambda x: x[1])]
        else:  # "first" or default
            # Process in order
            priorities = list(range(n_masks))
        
        # Assign pixels to masks based on priority
        for mask_idx in priorities:
            mask = masks[mask_idx]
            # Only assign pixels that aren't already owned
            unowned_pixels = (priority_map == -1) & mask
            priority_map[unowned_pixels] = mask_idx
        
        # Create resolved masks
        resolved_masks = np.zeros_like(masks)
        for i in range(n_masks):
            resolved_masks[i] = (priority_map == i)
        
        return resolved_masks

class BaseSam2SegmentationModel(BaseSegmentationModel):
    """
    Base class for segmentation models that use SAM2 for mask generation
    """
    
    def __init__(
        self,
        detector: BaseObjectDetector,
        sam2_model_id: str = "facebook/sam2.1-hiera-large",
        sam2_cfg_path: str = None,  # Deprecated
        sam2_ckpt_path: str = None,  # Deprecated
        device: str = "cuda",
        mask_threshold: float = 0.5,
        use_stability_score: bool = True,
        stability_score_thresh: float = 0.95,
        remove_small_regions: bool = True,
        min_region_area: int = 100,
        resolve_overlaps: bool = True,
        overlap_resolution: str = "score",
        mask_selection: str = "largest",
        clip_to_bbox: bool = True
    ):
        """
        Initialize base segmentation model
        
        Args:
            detector: Object detector instance
            sam2_model_id: HuggingFace model ID (e.g., "facebook/sam2.1-hiera-large")
            sam2_cfg_path: (Deprecated) Legacy parameter, ignored
            sam2_ckpt_path: (Deprecated) Legacy parameter, ignored
            device: Device to run models on
        """
        self.detector = detector
        self.device = device
        
        # Initialize SAM2 adapter with all parameters
        self.sam2_adapter = Sam2Adapter(
            sam2_model_id=sam2_model_id,
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            mask_threshold=mask_threshold,
            use_stability_score=use_stability_score,
            stability_score_thresh=stability_score_thresh,
            remove_small_regions=remove_small_regions,
            min_region_area=min_region_area,
            resolve_overlaps=resolve_overlaps,
            overlap_resolution=overlap_resolution,
            mask_selection=mask_selection,
            clip_to_bbox=clip_to_bbox
        )
    
    def segment(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]], Optional[np.ndarray]]:
        """
        Detect and segment objects in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes, class labels, confidence scores, segmentation masks)
        """
        # Check if file exists
        if not Path(image_path).exists():
            return None, None, None, None
            
        # Run detection
        boxes, labels, confidences = self.detector.detect(image_path)
        
        if boxes is None or len(boxes) == 0:
            return None, None, None, None
            
        # Load image for SAM2
        image = cv2.imread(str(image_path))
        if image is None:
            return boxes, labels, confidences, None
            
        # Generate precise masks with SAM2
        masks = self.sam2_adapter.generate_masks_from_boxes(image, boxes)
        
        return boxes, labels, confidences, masks

class YoloSam2SegmentationModel(BaseSam2SegmentationModel):
    """
    Segmentation model that combines YOLO for detection with SAM2 for precise segmentation
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        sam2_model_id: str = "facebook/sam2.1-hiera-large",
        sam2_cfg_path: str = None,  # Deprecated
        sam2_ckpt_path: str = None,  # Deprecated
        target_categories: Optional[List[str]] = None,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_yolov8: bool = False,
        mask_threshold: float = 0.5,
        use_stability_score: bool = True,
        stability_score_thresh: float = 0.95,
        remove_small_regions: bool = True,
        min_region_area: int = 100,
        resolve_overlaps: bool = True,
        overlap_resolution: str = "score",
        mask_selection: str = "largest",
        clip_to_bbox: bool = True
    ):
        """
        Initialize YOLO+SAM2 segmentation model
        
        Args:
            yolo_model_path: Path to YOLO model file
            sam2_model_id: HuggingFace model ID (e.g., "facebook/sam2.1-hiera-large")
            sam2_cfg_path: (Deprecated) Legacy parameter, ignored
            sam2_ckpt_path: (Deprecated) Legacy parameter, ignored
            target_categories: Optional list of target categories to filter detections
            device: Device to run models on
            conf_threshold: Confidence threshold for YOLO
            iou_threshold: IoU threshold for YOLO NMS
            use_yolov8: Whether to use YOLOv8 instead of YOLOv5
        """
        # Initialize YOLO detector
        detector = YOLODetectionModel(
            model_path=yolo_model_path,
            target_categories=target_categories if target_categories is not None else [],
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_yolov8=use_yolov8
        )
        
        # Initialize base class with all parameters
        super().__init__(
            detector=detector,
            sam2_model_id=sam2_model_id,
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            mask_threshold=mask_threshold,
            use_stability_score=use_stability_score,
            stability_score_thresh=stability_score_thresh,
            remove_small_regions=remove_small_regions,
            min_region_area=min_region_area,
            resolve_overlaps=resolve_overlaps,
            overlap_resolution=overlap_resolution,
            mask_selection=mask_selection,
            clip_to_bbox=clip_to_bbox
        )

class DinoSam2SegmentationModel(BaseSam2SegmentationModel):
    """
    Segmentation model that combines Grounding DINO for detection with SAM2 for precise segmentation
    """
    
    def __init__(
        self,
        text_prompt: str,
        sam2_model_id: str = "facebook/sam2.1-hiera-large",
        sam2_cfg_path: str = None,  # Deprecated
        sam2_ckpt_path: str = None,  # Deprecated
        grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        dino_cfg_path: str = None,  # Deprecated
        dino_ckpt_path: str = None,  # Deprecated
        mask_threshold: float = 0.5,
        use_stability_score: bool = True,
        stability_score_thresh: float = 0.95,
        remove_small_regions: bool = True,
        min_region_area: int = 100,
        resolve_overlaps: bool = True,
        overlap_resolution: str = "score",
        mask_selection: str = "largest",
        clip_to_bbox: bool = True
    ):
        """
        Initialize Grounding DINO+SAM2 segmentation model
        
        Args:
            text_prompt: Text prompt for object detection (comma-separated)
            sam2_model_id: HuggingFace model ID (e.g., "facebook/sam2.1-hiera-large")
            grounding_dino_model: HuggingFace model ID for Grounding DINO (e.g., "IDEA-Research/grounding-dino-tiny")
            device: Device to run models on
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Text similarity threshold
            Other args are SAM2 parameters for mask refinement
        """
        # Initialize DINO detector using Transformers
        if TRANSFORMERS_DINO_AVAILABLE:
            detector = TransformersDinoDetector(
                text_prompt=text_prompt,
                model_id=grounding_dino_model,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        else:
            raise ImportError(
                "Grounding DINO requires transformers library. "
                "Please install it with: pip install transformers"
            )
        
        # Initialize base class with all parameters
        super().__init__(
            detector=detector,
            sam2_model_id=sam2_model_id,
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            mask_threshold=mask_threshold,
            use_stability_score=use_stability_score,
            stability_score_thresh=stability_score_thresh,
            remove_small_regions=remove_small_regions,
            min_region_area=min_region_area,
            resolve_overlaps=resolve_overlaps,
            overlap_resolution=overlap_resolution,
            mask_selection=mask_selection,
            clip_to_bbox=clip_to_bbox
        )

class GenericInstanceSegmentationModel(BaseSegmentationModel):
    """
    Generic wrapper for any instance segmentation model.
    Simple and flexible - just pass your model and preprocessing/postprocessing functions.
    
    Example usage:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        def preprocess(image):
            # Convert to tensor and normalize
            return transform(image)
            
        def postprocess(outputs, image_shape):
            # Extract boxes, labels, scores, masks from model output
            return boxes, labels, scores, masks
            
        segmentation_model = GenericInstanceSegmentationModel(
            model=model,
            preprocess_fn=preprocess,
            postprocess_fn=postprocess,
            device='cuda'
        )
    """
    
    def __init__(
        self,
        model,
        preprocess_fn,
        postprocess_fn,
        device: str = "cuda",
        score_threshold: float = 0.5
    ):
        """
        Initialize generic instance segmentation wrapper.
        
        Args:
            model: Any pre-trained instance segmentation model
            preprocess_fn: Function to preprocess image before feeding to model
            postprocess_fn: Function to extract boxes, labels, scores, masks from model output
            device: Device to run model on
            score_threshold: Minimum score to keep predictions
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.device = device
        self.score_threshold = score_threshold
        
        # Move model to device and set to eval mode
        if hasattr(model, 'to'):
            self.model = self.model.to(device)
        if hasattr(model, 'eval'):
            self.model.eval()
    
    def segment(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]], Optional[np.ndarray]]:
        """
        Perform instance segmentation on an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (bounding boxes, class labels, confidence scores, segmentation masks)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        preprocessed = self.preprocess_fn(image_rgb)
        
        # Run model inference
        with torch.no_grad():
            if isinstance(preprocessed, list):
                # If it's a list of tensors, move each to device
                preprocessed = [t.to(self.device) if hasattr(t, 'to') else t for t in preprocessed]
                outputs = self.model(preprocessed)
            elif isinstance(preprocessed, torch.Tensor):
                # Move to device if it's a tensor
                preprocessed = preprocessed.to(self.device)
                outputs = self.model(preprocessed)
            else:
                outputs = self.model(preprocessed)
        
        # Postprocess outputs
        boxes, labels, scores, masks = self.postprocess_fn(outputs, image.shape[:2])
        
        # Filter by score threshold
        if scores is not None and len(scores) > 0:
            keep_indices = [i for i, score in enumerate(scores) if score >= self.score_threshold]
            if keep_indices:
                boxes = boxes[keep_indices] if boxes is not None else None
                labels = [labels[i] for i in keep_indices] if labels is not None else None
                scores = [scores[i] for i in keep_indices] if scores is not None else None
                masks = masks[keep_indices] if masks is not None else None
            else:
                return None, None, None, None
        
        return boxes, labels, scores, masks

class SegmentationBased:
    """Base class for segmentation-based manipulators"""
    
    def manipulate(self, 
                  image: np.ndarray, 
                  masks: np.ndarray, 
                  object_index: int,
                  preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Manipulate an image to hide/modify a specific object using segmentation masks,
        while preserving pixels from objects specified in preserve_indices.
        
        Args:
            image: Original image as numpy array
            masks: Array of segmentation masks with shape (N, H, W)
            object_index: Index of the object to manipulate
            preserve_indices: Optional list of object indices to preserve
            
        Returns:
            Manipulated image
        """
        raise NotImplementedError()
    
    def _create_preserve_mask(self, masks: np.ndarray, preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Create a mask of pixels that should be preserved (not modified).
        
        Args:
            masks: Array of segmentation masks with shape (N, H, W)
            preserve_indices: List of object indices to preserve
            
        Returns:
            Boolean mask where True indicates pixels to preserve
        """
        if not preserve_indices or len(preserve_indices) == 0:
            # No pixels to preserve
            return np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
        
        # Create a union of all masks we want to preserve
        preserve_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
        for idx in preserve_indices:
            if idx < len(masks):
                preserve_mask = preserve_mask | masks[idx].astype(bool)
                
        return preserve_mask


class BlackoutSegmentationManipulator(SegmentationBased):
    """Manipulator that blacks out an object using its segmentation mask or an alternate form"""
    
    def __init__(self, 
                expansion_pixels: int = 2, 
                mask_type: str = "precise", 
                coverage_factor: float = 1.3,
                max_complexity: int = 8,
                preserve_overlapping: bool = True):
        """
        Initialize blackout manipulator
        
        Args:
            expansion_pixels: Number of pixels to expand mask by for smoother edges
            mask_type: Type of mask to use:
                - "precise": Exact segmentation
                - "bbox": Full bounding box
                - "convex": Convex hull of the object
                - "blob": Approximated blob-like shape
                - "adaptive": Smart coverage that adapts to object shape
            coverage_factor: Factor to control how much extra area to cover (1.0 means exact, >1.0 means extra)
            max_complexity: Controls complexity for "adaptive" mask (higher = more detailed)
            preserve_overlapping: Whether to preserve pixels that belong to objects in preserve_indices
        """
        self.expansion_pixels = expansion_pixels
        self.mask_type = mask_type
        self.coverage_factor = coverage_factor
        self.max_complexity = max_complexity
        self.preserve_overlapping = preserve_overlapping
        
    def manipulate(self, 
                  image: np.ndarray, 
                  masks: np.ndarray, 
                  object_index: int,
                  preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Black out a specific object in the image using selected mask type,
        while optionally preserving pixels from objects specified in preserve_indices.
        
        Args:
            image: Original image as numpy array
            masks: Array of segmentation masks with shape (N, H, W)
            object_index: Index of the object to blackout
            preserve_indices: Optional list of object indices to preserve
            
        Returns:
            Image with specified object blacked out while preserving selected objects if preserve_overlapping is True
        """
        if object_index >= len(masks) or object_index < 0:
            return image.copy()
            
        result = image.copy()
        original_mask = masks[object_index].astype(np.uint8)
        
        # Generate appropriate mask based on mask_type
        if self.mask_type == "bbox":
            mask = self._get_bbox_mask(original_mask)
        elif self.mask_type == "convex":
            mask = self._get_convex_hull_mask(original_mask)
        elif self.mask_type == "blob":
            mask = self._get_blob_mask(original_mask)
        elif self.mask_type == "adaptive":
            mask = self._get_adaptive_mask(original_mask)
        else:  # "precise"
            mask = original_mask
            # Expand mask slightly for smoother edges
            if self.expansion_pixels > 0:
                kernel = np.ones((self.expansion_pixels, self.expansion_pixels), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Create a mask of pixels to preserve if needed
        if self.preserve_overlapping and preserve_indices and object_index in preserve_indices:
            # If we're trying to preserve the object we're manipulating, remove it from preserve_indices
            preserve_indices = [idx for idx in preserve_indices if idx != object_index]
        
        if self.preserve_overlapping and preserve_indices:
            preserve_mask = self._create_preserve_mask(masks, preserve_indices)
            # Apply mask only to pixels that aren't in the preserve_mask
            manipulation_mask = (mask > 0) & (~preserve_mask)
        else:
            # Apply mask to all pixels without considering preservation
            manipulation_mask = (mask > 0)
            
        result[manipulation_mask] = 0
        
        return result
    
    def _get_bbox_mask(self, mask):
        """Create a bounding box mask from segmentation mask"""
        import cv2
        import numpy as np
        
        # Compute bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return mask
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Expand bounding box if needed
        if self.expansion_pixels > 0:
            rmin = max(0, rmin - self.expansion_pixels)
            rmax = min(mask.shape[0] - 1, rmax + self.expansion_pixels)
            cmin = max(0, cmin - self.expansion_pixels)
            cmax = min(mask.shape[1] - 1, cmax + self.expansion_pixels)
        
        # Create a new mask from bounding box
        bbox_mask = np.zeros_like(mask)
        bbox_mask[rmin:rmax+1, cmin:cmax+1] = 1
        return bbox_mask

    def _get_convex_hull_mask(self, mask):
        """Create a convex hull mask from segmentation mask"""
        import cv2
        import numpy as np
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Create a mask from the hull
        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], 0, 1, -1)
        
        # Optionally expand the mask
        if self.expansion_pixels > 0:
            kernel = np.ones((self.expansion_pixels, self.expansion_pixels), np.uint8)
            hull_mask = cv2.dilate(hull_mask, kernel, iterations=1)
            
        return hull_mask
        
    def _get_blob_mask(self, mask):
        """Create a blob-like mask that is simpler than the original but still covers it"""
        import cv2
        import numpy as np
        
        # Apply morphological operations to simplify
        kernel_size = max(5, int(min(mask.shape) * 0.03))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Close the mask to fill small holes
        blob = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Open the mask to remove small protrusions
        blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
        
        # Dilate to ensure coverage
        expand_factor = int(kernel_size * (self.coverage_factor - 1))
        if expand_factor > 0:
            expand_kernel = np.ones((expand_factor, expand_factor), np.uint8)
            blob = cv2.dilate(blob, expand_kernel, iterations=1)
            
        return blob
    
    def _get_adaptive_mask(self, mask):
        """
        Create an adaptive mask that balances between coverage and shape.
        This creates a simplified polygon that adapts to the object shape.
        """
        import cv2
        import numpy as np
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the original area for comparison
        original_area = cv2.contourArea(largest_contour)
        
        # Create an adaptive approximation based on object size and our parameters
        # Adjust epsilon based on perimeter and our max_complexity setting
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = perimeter / max(4, min(20, self.max_complexity))
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Calculate area ratio to original
        approx_area = cv2.contourArea(approx)
        area_ratio = approx_area / original_area if original_area > 0 else 1.0
        
        # If our approximation is too small, try a convex hull instead
        if area_ratio < 0.9:  # If we're missing >10% of the original area
            approx = cv2.convexHull(largest_contour)
            
        # Create a mask from the approximated polygon
        adaptive_mask = np.zeros_like(mask)
        cv2.drawContours(adaptive_mask, [approx], 0, 1, -1)
        
        # Scale the contour to ensure it covers the original plus some margin
        if self.coverage_factor > 1.0:
            # Find the centroid of the contour
            M = cv2.moments(approx)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Scale the contour points from the centroid
                scaled_approx = approx.copy().astype(np.float32)
                for i in range(len(scaled_approx)):
                    scaled_approx[i][0][0] = cx + (scaled_approx[i][0][0] - cx) * self.coverage_factor
                    scaled_approx[i][0][1] = cy + (scaled_approx[i][0][1] - cy) * self.coverage_factor
                
                # Convert back to int for drawing
                scaled_approx = scaled_approx.astype(np.int32)
                
                # Draw the scaled contour
                adaptive_mask = np.zeros_like(mask)
                cv2.drawContours(adaptive_mask, [scaled_approx], 0, 1, -1)
                
        # Ensure we're not going outside the image
        adaptive_mask = np.clip(adaptive_mask, 0, 1)
        
        # Optionally expand the mask slightly for smoother edges
        if self.expansion_pixels > 0:
            kernel = np.ones((self.expansion_pixels, self.expansion_pixels), np.uint8)
            adaptive_mask = cv2.dilate(adaptive_mask, kernel, iterations=1)
            
        return adaptive_mask
    
    def _create_preserve_mask(self, masks: np.ndarray, preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Create a mask of pixels that should be preserved (not modified).
        
        Args:
            masks: Array of segmentation masks with shape (N, H, W)
            preserve_indices: List of object indices to preserve
            
        Returns:
            Boolean mask where True indicates pixels to preserve
        """
        if not preserve_indices or len(preserve_indices) == 0:
            # No pixels to preserve
            return np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
        
        # Create a union of all masks we want to preserve
        preserve_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
        for idx in preserve_indices:
            if idx < len(masks):
                preserve_mask = preserve_mask | masks[idx].astype(bool)
                
        return preserve_mask


class BlurSegmentationManipulator(SegmentationBased):
    """Manipulator that blurs an object using its segmentation mask or bounding box"""
    
    def __init__(self, blur_strength: int = 25, expansion_pixels: int = 2, use_bbox: bool = False):
        """
        Initialize blur manipulator
        
        Args:
            blur_strength: Strength of Gaussian blur (odd number)
            expansion_pixels: Number of pixels to expand mask by for smoother edges
            use_bbox: If True, use bounding box instead of precise segmentation
        """
        # Ensure blur_strength is odd
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self.expansion_pixels = expansion_pixels
        self.use_bbox = use_bbox
        
    def manipulate(self, 
                  image: np.ndarray, 
                  masks: np.ndarray, 
                  object_index: int,
                  preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Blur a specific object in the image using its segmentation mask or bounding box,
        while preserving pixels from objects specified in preserve_indices.
        
        Args:
            image: Original image as numpy array
            masks: Array of segmentation masks with shape (N, H, W)
            object_index: Index of the object to blur
            preserve_indices: Optional list of object indices to preserve
            
        Returns:
            Image with specified object blurred
        """
        if object_index >= len(masks):
            return image.copy()
            
        result = image.copy()
        mask = masks[object_index].astype(np.uint8)
        
        if self.use_bbox:
            # Compute bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Expand bounding box if needed
            if self.expansion_pixels > 0:
                rmin = max(0, rmin - self.expansion_pixels)
                rmax = min(mask.shape[0] - 1, rmax + self.expansion_pixels)
                cmin = max(0, cmin - self.expansion_pixels)
                cmax = min(mask.shape[1] - 1, cmax + self.expansion_pixels)
            
            # Create a new mask from bounding box
            bbox_mask = np.zeros_like(mask)
            bbox_mask[rmin:rmax+1, cmin:cmax+1] = 1
            mask = bbox_mask
        else:
            # Optional: Expand mask slightly for smoother edges
            if self.expansion_pixels > 0:
                kernel = np.ones((self.expansion_pixels, self.expansion_pixels), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Create a mask of pixels to preserve
        if preserve_indices and object_index in preserve_indices:
            # If we're trying to preserve the object we're manipulating, remove it from preserve_indices
            preserve_indices = [idx for idx in preserve_indices if idx != object_index]
            
        preserve_mask = self._create_preserve_mask(masks, preserve_indices)
        
        # Create blurred version of entire image
        blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)
        
        # Apply mask to replace original pixels with blurred pixels, but only where not in preserve_mask
        manipulation_mask = (mask > 0) & (~preserve_mask)
        manipulation_mask_3d = np.stack([manipulation_mask] * 3, axis=2)
        result = np.where(manipulation_mask_3d, blurred, result)
        
        return result


class InpaintSegmentationManipulator(SegmentationBased):
    """Manipulator that inpaints (removes and fills) an object using its segmentation mask or bounding box"""
    
    def __init__(self, expansion_pixels: int = 3, inpaint_radius: int = 10, use_bbox: bool = False):
        """
        Initialize inpainting manipulator
        
        Args:
            expansion_pixels: Number of pixels to expand mask by
            inpaint_radius: Radius of neighborhood for inpainting
            use_bbox: If True, use bounding box instead of precise segmentation
        """
        self.expansion_pixels = expansion_pixels
        self.inpaint_radius = inpaint_radius
        self.use_bbox = use_bbox
        
    def manipulate(self, 
                  image: np.ndarray, 
                  masks: np.ndarray, 
                  object_index: int,
                  preserve_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Inpaint (remove and fill) a specific object in the image while preserving 
        pixels from objects specified in preserve_indices.
        
        Args:
            image: Original image as numpy array
            masks: Array of segmentation masks with shape (N, H, W)
            object_index: Index of the object to inpaint
            preserve_indices: Optional list of object indices to preserve
            
        Returns:
            Image with specified object inpainted
        """
        if object_index >= len(masks):
            return image.copy()
            
        # Convert image to BGR if it's RGB (cv2.inpaint expects BGR)
        if image.shape[2] == 3 and image[0,0,2] < image[0,0,0]:  # Simple RGB check
            img_for_inpaint = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_for_inpaint = image.copy()
            
        mask = masks[object_index].astype(np.uint8)
        
        if self.use_bbox:
            # Compute bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Expand bounding box if needed
            if self.expansion_pixels > 0:
                rmin = max(0, rmin - self.expansion_pixels)
                rmax = min(mask.shape[0] - 1, rmax + self.expansion_pixels)
                cmin = max(0, cmin - self.expansion_pixels)
                cmax = min(mask.shape[1] - 1, cmax + self.expansion_pixels)
            
            # Create a new mask from bounding box
            bbox_mask = np.zeros_like(mask)
            bbox_mask[rmin:rmax+1, cmin:cmax+1] = 1
            mask = bbox_mask
        else:
            # Optional: Expand mask slightly
            if self.expansion_pixels > 0:
                kernel = np.ones((self.expansion_pixels, self.expansion_pixels), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Create a mask of pixels to preserve
        if preserve_indices and object_index in preserve_indices:
            # If we're trying to preserve the object we're manipulating, remove it from preserve_indices
            preserve_indices = [idx for idx in preserve_indices if idx != object_index]
            
        preserve_mask = self._create_preserve_mask(masks, preserve_indices)
        
        # Only inpaint pixels that are in the mask AND not in preserve_mask
        inpaint_mask = (mask > 0) & (~preserve_mask)
        inpaint_mask = inpaint_mask.astype(np.uint8) * 255
        
        # Only perform inpainting if there are pixels to inpaint
        if np.any(inpaint_mask):
            # Inpaint
            result = cv2.inpaint(img_for_inpaint, inpaint_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        else:
            result = img_for_inpaint
        
        # Convert back to RGB if needed
        if image.shape[2] == 3 and image[0,0,2] < image[0,0,0]:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
        return result