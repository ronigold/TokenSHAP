#image_utils.py

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import cv2
from abc import ABC, abstractmethod
import torch
from torchvision.ops import box_convert
import torchvision.transforms as transforms

# Import computer vision related libraries
try:
    import sys, os
    sys.path.append(os.environ["DYNAMIC_SAM2_PATH"])

    from dynamic_sam2.sam2_video_tracker import Sam2VideoTracker
    from dynamic_sam2.object_detection import YOLODetectionModel, DinoDetectionModel
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.models.GroundingDINO.ms_deform_attn import (
        multi_scale_deformable_attn_pytorch,
        MultiScaleDeformableAttnFunction
    )
except ImportError:
    print("Warning: Some optional dependencies (sam2, grounding_dino) not installed, Please provide your own segmentation model!")

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
    """Adapter for SAM2 segmentation model"""
    
    def __init__(
        self,
        sam2_cfg_path: str,
        sam2_ckpt_path: str,
        device: str = "cuda"
    ):
        """
        Initialize SAM2 adapter
        
        Args:
            sam2_cfg_path: Path to SAM2 config file
            sam2_ckpt_path: Path to SAM2 checkpoint file
            device: Device to run model on
        """
        self.device = device
        self.sam2_cfg_path = sam2_cfg_path
        self.sam2_ckpt_path = sam2_ckpt_path
        self._initialize_sam2()
        
    def _initialize_sam2(self):
        """Initialize SAM2 model"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Fix for missing _C module in MultiScaleDeformableAttention if needed
            try:
                from grounding_dino.groundingdino.models.GroundingDINO.ms_deform_attn import (
                    multi_scale_deformable_attn_pytorch, 
                    MultiScaleDeformableAttnFunction
                )

                # Create replacement for MultiScaleDeformableAttnFunction.apply if needed
                if not hasattr(MultiScaleDeformableAttnFunction, 'apply'):
                    class DummyFunction:
                        @staticmethod
                        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                            return multi_scale_deformable_attn_pytorch(
                                value, spatial_shapes, sampling_locations, attention_weights
                            )
                    MultiScaleDeformableAttnFunction.apply = DummyFunction.apply
            except ImportError:
                # The fix is only needed if using groundingdino with SAM2
                pass
                
            sam2_model = build_sam2(self.sam2_cfg_path, self.sam2_ckpt_path)
            self.image_predictor = SAM2ImagePredictor(sam2_model)
        except ImportError as e:
            raise ImportError(f"SAM2 dependencies not installed: {str(e)}")
    
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
            
        self.image_predictor.set_image(image_rgb)
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        
        if masks is None:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            
        # Ensure correct shape
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        elif masks.ndim == 2:
            masks = masks[np.newaxis, ...]
            
        return masks.astype(bool)

class BaseSam2SegmentationModel(BaseSegmentationModel):
    """
    Base class for segmentation models that use SAM2 for mask generation
    """
    
    def __init__(
        self,
        detector: BaseObjectDetector,
        sam2_cfg_path: str,
        sam2_ckpt_path: str,
        device: str = "cuda"
    ):
        """
        Initialize base segmentation model
        
        Args:
            detector: Object detector instance
            sam2_cfg_path: Path to SAM2 config file
            sam2_ckpt_path: Path to SAM2 checkpoint file
            device: Device to run models on
        """
        self.detector = detector
        self.device = device
        
        # Initialize SAM2 adapter
        self.sam2_adapter = Sam2Adapter(
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device
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
        sam2_cfg_path: str,
        sam2_ckpt_path: str,
        target_categories: Optional[List[str]] = None,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_yolov8: bool = False
    ):
        """
        Initialize YOLO+SAM2 segmentation model
        
        Args:
            yolo_model_path: Path to YOLO model file
            sam2_cfg_path: Path to SAM2 config file
            sam2_ckpt_path: Path to SAM2 checkpoint file
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
        
        # Initialize base class
        super().__init__(
            detector=detector,
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device
        )

class DinoSam2SegmentationModel(BaseSam2SegmentationModel):
    """
    Segmentation model that combines Grounding DINO for detection with SAM2 for precise segmentation
    """
    
    def __init__(
        self,
        text_prompt: str,
        sam2_cfg_path: str,
        sam2_ckpt_path: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        dino_cfg_path: str = "../grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        dino_ckpt_path: str = "../gdino_checkpoints/groundingdino_swint_ogc.pth"
    ):
        """
        Initialize Grounding DINO+SAM2 segmentation model
        
        Args:
            text_prompt: Text prompt for object detection
            sam2_cfg_path: Path to SAM2 config file
            sam2_ckpt_path: Path to SAM2 checkpoint file
            device: Device to run models on
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Text similarity threshold
            dino_cfg_path: Path to Grounding DINO config file
            dino_ckpt_path: Path to Grounding DINO checkpoint file
        """
        # Initialize DINO detector
        detector = DinoDetectionModel(
            text_prompt=text_prompt,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            dino_cfg_path=dino_cfg_path,
            dino_ckpt_path=dino_ckpt_path
        )
        
        # Initialize base class
        super().__init__(
            detector=detector,
            sam2_cfg_path=sam2_cfg_path,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device
        )

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