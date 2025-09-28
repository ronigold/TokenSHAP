"""
Chunked Video Tracking with SAM2 Transformers
Memory-efficient implementation that processes video in chunks to prevent kernel crashes
"""

import logging
import torch
import cv2
import numpy as np
import os
import gc
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
from dataclasses import dataclass, field
from transformers import Sam2VideoModel, Sam2VideoProcessor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrackedObject:
    """Data class for tracked objects with proper attribute access"""
    object_id: int
    label: str
    frames: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> bbox
    masks: Dict[int, np.ndarray] = field(default_factory=dict)   # frame_idx -> mask
    confidences: Dict[int, float] = field(default_factory=dict)  # frame_idx -> confidence
    first_frame: int = 0
    last_frame: int = 0
    
    def add_detection(self, frame_idx: int, bbox: np.ndarray, confidence: float = 1.0, mask: Optional[np.ndarray] = None):
        """Add a detection to this tracked object"""
        self.frames[frame_idx] = bbox
        self.confidences[frame_idx] = confidence
        if mask is not None:
            self.masks[frame_idx] = mask
        
        # Update frame range
        if not self.frames or frame_idx < self.first_frame:
            self.first_frame = frame_idx
        if not self.frames or frame_idx > self.last_frame:
            self.last_frame = frame_idx
    
    @property
    def num_frames(self) -> int:
        """Number of frames this object appears in"""
        return len(self.frames)


class ChunkedSam2VideoTracker:
    """
    Memory-efficient video tracker using SAM2 from Transformers.
    Processes video in chunks to prevent memory crashes.
    """
    
    def __init__(
        self,
        video_path: str,
        detection_model,
        output_dir: str = "tracking_results",
        device: str = "cuda",
        chunk_size: int = 30,  # Process 30 frames at a time
        overlap_frames: int = 5,  # Overlap between chunks for continuity
        check_interval: int = 5,  # Check for new objects every N frames
        sam2_model_id: str = "facebook/sam2.1-hiera-small",  # Use smaller model
        iou_threshold: float = 0.5,
        min_tracked_frames: int = 5,
        generate_video: bool = True,
        save_masks: bool = False,
        max_frames: Optional[int] = None,
        memory_limit_gb: float = 4.0,
        verbose: bool = True
    ):
        """
        Initialize chunked video tracker.
        
        Args:
            video_path: Path to input video
            detection_model: Object detection model (YOLO, etc.)
            chunk_size: Number of frames to process at once
            overlap_frames: Frames to overlap between chunks
            check_interval: Frames between new object detection
            sam2_model_id: HuggingFace model ID
            memory_limit_gb: GPU memory limit in GB
        """
        self.video_path = Path(video_path)
        self.detection_model = detection_model
        self.output_dir = Path(output_dir)
        self.device = device
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.check_interval = check_interval
        self.iou_threshold = iou_threshold
        self.min_tracked_frames = min_tracked_frames
        self.generate_video = generate_video
        self.save_masks = save_masks
        self.max_frames = max_frames
        self.memory_limit_gb = memory_limit_gb
        self.verbose = verbose
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize tracking data
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.sam2_id_mapping = {}  # Maps SAM2 IDs to our tracked object IDs
        
        # Load SAM2 model with memory efficiency
        self._load_sam2_model(sam2_model_id)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('ChunkedSam2VideoTracker')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "tracker.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_sam2_model(self, model_id: str):
        """Load SAM2 model - NO FALLBACKS! SAM2 IS FULLY SUPPORTED!"""
        self.logger.info(f"Loading SAM2 Model: {model_id}")
        self.logger.info("SAM2 IS FULLY SUPPORTED IN THIS ENVIRONMENT!")
        
        # Clear memory before loading
        self._clear_memory()
        
        # Use appropriate dtype for memory efficiency
        if self.device == "cuda" and torch.cuda.is_available():
            # Use bfloat16 for newer GPUs, float16 for older ones
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] >= 8:  # Ampere and newer
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Load REAL SAM2 model - IT EXISTS!
        self.sam2_model = Sam2VideoModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.sam2_model.eval()
        self.sam2_processor = Sam2VideoProcessor.from_pretrained(model_id)
        
        self.logger.info("✅ SAM2 model loaded successfully - REAL SAM2, NOT A MOCK!")
    
    def _clear_memory(self):
        """Clear GPU/CPU memory"""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        if self.device == "cuda" and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_limit = self.memory_limit_gb
            
            if memory_used > memory_limit * 0.9:  # 90% threshold
                self.logger.warning(f"Memory usage high: {memory_used:.2f}GB / {memory_limit:.2f}GB")
                self._clear_memory()
                return False
        return True
    
    def _extract_frames(self) -> Tuple[int, float, int, int]:
        """Extract frames from video to disk"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Limit frames if specified
        if self.max_frames:
            total_frames = min(total_frames, self.max_frames)
        
        self.video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height
        }
        
        self.logger.info(f"Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
        
        # Extract frames
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame
            frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            frame_idx += 1
            
            # Clear memory periodically
            if frame_idx % 100 == 0:
                self.logger.debug(f"Extracted {frame_idx}/{total_frames} frames")
        
        cap.release()
        self.logger.info(f"Extracted {frame_idx} frames")
        return frame_idx, fps, width, height
    
    def _load_chunk_frames(self, start_idx: int, end_idx: int) -> List[Image.Image]:
        """Load a chunk of frames as PIL images"""
        frames = []
        for idx in range(start_idx, end_idx + 1):
            frame_path = self.frames_dir / f"frame_{idx:06d}.jpg"
            if frame_path.exists():
                # Load as PIL image directly for SAM2
                pil_image = Image.open(frame_path).convert("RGB")
                frames.append(pil_image)
        return frames
    
    def _detect_objects_in_frame(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[float]]]:
        """Run object detection on a frame"""
        frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
        
        try:
            if hasattr(self.detection_model, 'detect'):
                # Custom detection model
                return self.detection_model.detect(str(frame_path))
            elif hasattr(self.detection_model, 'predict'):
                # YOLO model
                results = self.detection_model.predict(str(frame_path), verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        labels = [result.names[int(c)] for c in result.boxes.cls]
                        confidences = result.boxes.conf.cpu().numpy().tolist()
                        return boxes, labels, confidences
            
            return None, None, None
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return None, None, None
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detection_to_tracked(self, box: np.ndarray, frame_idx: int) -> Optional[int]:
        """Match a detection to existing tracked objects"""
        best_match_id = None
        best_iou = self.iou_threshold
        
        # Look for matches in recent frames
        for obj_id, tracked_obj in self.tracked_objects.items():
            # Check last few frames for this object
            for check_frame in range(max(0, frame_idx - 5), frame_idx):
                if check_frame in tracked_obj.frames:
                    existing_box = tracked_obj.frames[check_frame]
                    iou = self._calculate_iou(box, existing_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = obj_id
                        break
        
        return best_match_id
    
    def _process_chunk(self, chunk_start: int, chunk_end: int, carry_over_tracks: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Process a chunk of frames with SAM2.
        
        Args:
            chunk_start: Start frame index
            chunk_end: End frame index (inclusive)
            carry_over_tracks: Tracks from previous chunk to continue
            
        Returns:
            Tracks to carry over to next chunk
        """
        self.logger.info(f"Processing chunk: frames {chunk_start} to {chunk_end}")
        
        # Load chunk frames
        chunk_frames = self._load_chunk_frames(chunk_start, chunk_end)
        if not chunk_frames:
            self.logger.warning(f"No frames loaded for chunk {chunk_start}-{chunk_end}")
            return {}
        
        # Initialize SAM2 video session for this chunk
        try:
            inference_session = self.sam2_processor.init_video_session(
                video=chunk_frames,
                inference_device=self.device,
                dtype=self.dtype,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2 session: {e}")
            self._clear_memory()
            return {}
        
        # Track SAM2 IDs for this chunk
        chunk_sam2_mapping = {}
        next_sam2_id = 0
        
        # Collect objects to add (will process sequentially with boxes)
        objects_to_add = []  # List of (sam2_id, tracked_id, box)
        
        # Collect carry-over tracks
        for tracked_id, track_info in carry_over_tracks.items():
            if 'box' in track_info:
                box = track_info['box']
                sam2_id = next_sam2_id
                next_sam2_id += 1
                
                if isinstance(box, np.ndarray):
                    box_list = box.tolist()
                else:
                    box_list = list(box)
                
                objects_to_add.append((sam2_id, tracked_id, box_list))
                chunk_sam2_mapping[sam2_id] = tracked_id
        
        # Detect new objects at the start of the chunk if needed
        if chunk_start % self.check_interval == 0:
            boxes, labels, confidences = self._detect_objects_in_frame(chunk_start)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Check if this matches existing object
                    matched_id = self._match_detection_to_tracked(box, chunk_start)
                    
                    if matched_id is None:
                        # New object detected
                        obj_id = self.next_object_id
                        self.next_object_id += 1
                        
                        # Create tracked object
                        new_obj = TrackedObject(
                            object_id=obj_id,
                            label=labels[i] if i < len(labels) else "object",
                            first_frame=chunk_start,
                            last_frame=chunk_start
                        )
                        
                        conf = confidences[i] if confidences and i < len(confidences) else 1.0
                        new_obj.add_detection(chunk_start, box, conf)
                        
                        self.tracked_objects[obj_id] = new_obj
                        
                        # Collect for sequential adding to SAM2
                        sam2_id = next_sam2_id
                        next_sam2_id += 1
                        
                        if isinstance(box, np.ndarray):
                            box_list = box.tolist()
                        else:
                            box_list = list(box)
                        
                        objects_to_add.append((sam2_id, obj_id, box_list))
                        chunk_sam2_mapping[sam2_id] = obj_id
                        
                        self.logger.info(f"New object {obj_id} ({new_obj.label}) at frame {chunk_start}")
                    else:
                        # Update existing object
                        conf = confidences[i] if confidences and i < len(confidences) else 1.0
                        self.tracked_objects[matched_id].add_detection(chunk_start, box, conf)
        
        # Add objects to SAM2 sequentially with boxes and initialize each
        if objects_to_add:
            self.logger.info(f"Adding {len(objects_to_add)} objects to SAM2 with boxes")
            
            for sam2_id, tracked_id, box in objects_to_add:
                # Add this object's box
                self.sam2_processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=0,
                    obj_ids=sam2_id,
                    input_boxes=[[box]],  # Single box with 3 levels
                )
                self.logger.debug(f"Added object SAM2 ID {sam2_id} (tracked ID {tracked_id}) with box {box}")
                
                # Process frame 0 to initialize memory for this object
                with torch.no_grad():
                    outputs = self.sam2_model(
                        inference_session=inference_session,
                        frame_idx=0
                    )
                    
                    # Mark this frame as tracked for each object
                    for idx, sid in enumerate(inference_session.obj_ids):
                        if sid == sam2_id:
                            if idx not in inference_session.frames_tracked_per_obj:
                                inference_session.frames_tracked_per_obj[idx] = {}
                            inference_session.frames_tracked_per_obj[idx][0] = True
                            self.logger.debug(f"Initialized memory for object {sam2_id} at index {idx}")
                            break
            
            self.logger.info(f"Successfully added and initialized {len(objects_to_add)} objects")
        
        # Process masks from initialization to extract bounding boxes
        if chunk_sam2_mapping and objects_to_add:
            # Get the final masks after all objects are added
            with torch.no_grad():
                outputs = self.sam2_model(
                    inference_session=inference_session,
                    frame_idx=0
                )
                
                # Process the masks to extract bounding boxes
                if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                    self.logger.debug(f"Processing {outputs.pred_masks.shape[0]} masks from frame 0")
                    for obj_idx, mask in enumerate(outputs.pred_masks):
                        if obj_idx < len(inference_session.obj_ids):
                            sam2_id = inference_session.obj_ids[obj_idx]
                            if sam2_id in chunk_sam2_mapping:
                                tracked_id = chunk_sam2_mapping[sam2_id]
                                # Convert mask to numpy and extract bbox
                                mask_np = mask.cpu().float().numpy() if isinstance(mask, torch.Tensor) else mask
                                if mask_np.ndim > 2:
                                    mask_np = mask_np.squeeze()
                                if mask_np.any():
                                    coords = np.where(mask_np > 0.5)
                                    if len(coords[0]) > 0:
                                        # Get bounds in mask space
                                        y_min_mask, y_max_mask = coords[0].min(), coords[0].max()
                                        x_min_mask, x_max_mask = coords[1].min(), coords[1].max()
                                        
                                        # Scale to original image space
                                        mask_h, mask_w = mask_np.shape
                                        scale_x = self.video_info['width'] / mask_w
                                        scale_y = self.video_info['height'] / mask_h
                                        
                                        x_min = int(x_min_mask * scale_x)
                                        y_min = int(y_min_mask * scale_y)
                                        x_max = int(x_max_mask * scale_x)
                                        y_max = int(y_max_mask * scale_y)
                                        
                                        box = np.array([x_min, y_min, x_max, y_max])
                                        # Add detection for annotation frame
                                        if tracked_id in self.tracked_objects:
                                            abs_frame_idx = chunk_start
                                            self.tracked_objects[tracked_id].add_detection(
                                                abs_frame_idx, box, 1.0,
                                                mask_np if self.save_masks else None
                                            )
        
        # Propagate masks through remaining frames - REAL SAM2 ONLY, NO FALLBACKS!
        if chunk_sam2_mapping and len(chunk_frames) > 1:
            self.logger.info(f"Propagating {len(chunk_sam2_mapping)} objects through remaining {len(chunk_frames)-1} frames")
            
            with torch.no_grad():
                # NO TRY-EXCEPT! If SAM2 fails, let it crash!
                # Use the iterator to propagate through all frames
                # Note: propagate_in_video_iterator will handle all frames including frame 0
                for sam2_output in self.sam2_model.propagate_in_video_iterator(
                    inference_session,
                    start_frame_idx=0,  # Start from beginning
                    max_frame_num_to_track=len(chunk_frames)
                ):
                    rel_frame_idx = sam2_output.frame_idx
                    abs_frame_idx = chunk_start + rel_frame_idx
                    
                    # Skip frame 0 as we already processed it during initialization
                    if rel_frame_idx == 0:
                        continue
                    
                    # Check for new objects periodically
                    if abs_frame_idx % self.check_interval == 0 and abs_frame_idx != chunk_start:
                        self.logger.debug(f"Checking for new objects at frame {abs_frame_idx}")
                        
                        # Detect objects in current frame
                        new_boxes, new_labels, new_confidences = self._detect_objects_in_frame(abs_frame_idx)
                        
                        if new_boxes is not None:
                            new_objects_to_add = []
                            
                            for i, box in enumerate(new_boxes):
                                # Check if this matches existing object
                                matched_id = self._match_detection_to_tracked(box, abs_frame_idx)
                                
                                if matched_id is None:
                                    # New object detected mid-chunk
                                    obj_id = self.next_object_id
                                    self.next_object_id += 1
                                    
                                    # Create tracked object
                                    new_obj = TrackedObject(
                                        object_id=obj_id,
                                        label=new_labels[i] if i < len(new_labels) else "object",
                                        first_frame=abs_frame_idx,
                                        last_frame=abs_frame_idx
                                    )
                                    
                                    conf = new_confidences[i] if new_confidences and i < len(new_confidences) else 1.0
                                    new_obj.add_detection(abs_frame_idx, box, conf)
                                    
                                    self.tracked_objects[obj_id] = new_obj
                                    
                                    # Prepare to add to SAM2
                                    sam2_id = len(inference_session.obj_ids)  # Next available SAM2 ID
                                    
                                    if isinstance(box, np.ndarray):
                                        box_list = box.tolist()
                                    else:
                                        box_list = list(box)
                                    
                                    new_objects_to_add.append((sam2_id, obj_id, box_list))
                                    chunk_sam2_mapping[sam2_id] = obj_id
                                    
                                    self.logger.info(f"New object {obj_id} ({new_obj.label}) detected at frame {abs_frame_idx}")
                                else:
                                    # Update existing object position from YOLO
                                    conf = new_confidences[i] if new_confidences and i < len(new_confidences) else 1.0
                                    self.tracked_objects[matched_id].add_detection(abs_frame_idx, box, conf)
                            
                            # Add new objects to current SAM2 session
                            if new_objects_to_add:
                                self.logger.info(f"Adding {len(new_objects_to_add)} new objects at frame {abs_frame_idx}")
                                
                                for sam2_id, tracked_id, box in new_objects_to_add:
                                    # Add this object's box at current frame
                                    self.sam2_processor.add_inputs_to_inference_session(
                                        inference_session=inference_session,
                                        frame_idx=rel_frame_idx,  # Use relative frame index
                                        obj_ids=sam2_id,
                                        input_boxes=[[box]],
                                    )
                                    
                                    # Process this frame to initialize the new object
                                    outputs_new = self.sam2_model(
                                        inference_session=inference_session,
                                        frame_idx=rel_frame_idx
                                    )
                                    
                                    # Mark frame as tracked for new object
                                    for idx, sid in enumerate(inference_session.obj_ids):
                                        if sid == sam2_id:
                                            if idx not in inference_session.frames_tracked_per_obj:
                                                inference_session.frames_tracked_per_obj[idx] = {}
                                            inference_session.frames_tracked_per_obj[idx][rel_frame_idx] = True
                                            break
                                
                                self.logger.info(f"Successfully added {len(new_objects_to_add)} new objects")
                    
                    # Process masks for this frame
                    if hasattr(sam2_output, 'pred_masks') and sam2_output.pred_masks is not None:
                        masks = sam2_output.pred_masks
                        
                        # Process each tracked object's mask
                        for obj_idx, mask in enumerate(masks):
                            if obj_idx < len(inference_session.obj_ids):
                                sam2_id = inference_session.obj_ids[obj_idx]
                                
                                if sam2_id in chunk_sam2_mapping:
                                    tracked_id = chunk_sam2_mapping[sam2_id]
                                    
                                    # Convert mask to numpy
                                    if isinstance(mask, torch.Tensor):
                                        mask_np = mask.cpu().float().numpy()  # Convert to float first for bfloat16
                                    else:
                                        mask_np = mask
                                    
                                    # Remove extra dimensions if present
                                    if mask_np.ndim > 2:
                                        mask_np = mask_np.squeeze()
                                    
                                    # Extract bounding box from mask
                                    if mask_np.any():
                                        coords = np.where(mask_np > 0.5)
                                        if len(coords[0]) > 0:
                                            # Get bounds in mask space
                                            y_min_mask, y_max_mask = coords[0].min(), coords[0].max()
                                            x_min_mask, x_max_mask = coords[1].min(), coords[1].max()
                                            
                                            # Scale to original image space
                                            mask_h, mask_w = mask_np.shape
                                            scale_x = self.video_info['width'] / mask_w
                                            scale_y = self.video_info['height'] / mask_h
                                            
                                            x_min = int(x_min_mask * scale_x)
                                            y_min = int(y_min_mask * scale_y)
                                            x_max = int(x_max_mask * scale_x)
                                            y_max = int(y_max_mask * scale_y)
                                            
                                            box = np.array([x_min, y_min, x_max, y_max])
                                            
                                            # Update tracked object
                                            if tracked_id in self.tracked_objects:
                                                self.tracked_objects[tracked_id].add_detection(
                                                    abs_frame_idx, box, 1.0,
                                                    mask_np if self.save_masks else None
                                                )
                    
                    # Log progress
                    if rel_frame_idx % 10 == 0:
                        self.logger.debug(f"Propagated to frame {rel_frame_idx}/{len(chunk_frames)}")
                        self._check_memory()
                
                self.logger.info(f"Successfully propagated through {len(chunk_frames)} frames")
        
        # Prepare carry-over tracks for next chunk
        carry_over = {}
        for sam2_id, tracked_id in chunk_sam2_mapping.items():
            if tracked_id in self.tracked_objects:
                obj = self.tracked_objects[tracked_id]
                # Get last known position
                if obj.frames:
                    last_frame = max(obj.frames.keys())
                    if last_frame >= chunk_end - self.overlap_frames:
                        carry_over[tracked_id] = {
                            'box': obj.frames[last_frame],
                            'label': obj.label
                        }
        
        # Clean up session
        del inference_session
        self._clear_memory()
        
        return carry_over
    
    def process_video(self) -> Dict[int, TrackedObject]:
        """
        Process video in chunks with memory management.
        
        Returns:
            Dictionary of tracked objects
        """
        try:
            # Extract frames
            self.logger.info("Extracting frames from video...")
            num_frames, fps, width, height = self._extract_frames()
            
            if num_frames == 0:
                self.logger.error("No frames extracted")
                return {}
            
            # Process video in chunks
            self.logger.info(f"Processing {num_frames} frames in chunks of {self.chunk_size}")
            
            chunk_start = 0
            carry_over_tracks = {}
            
            while chunk_start < num_frames:
                # Calculate chunk boundaries
                chunk_end = min(chunk_start + self.chunk_size - 1, num_frames - 1)
                
                # Check memory before processing chunk
                if not self._check_memory():
                    self.logger.warning(f"Stopping at frame {chunk_start} due to memory limit")
                    break
                
                # Process chunk
                carry_over_tracks = self._process_chunk(chunk_start, chunk_end, carry_over_tracks)
                
                # Move to next chunk with overlap
                chunk_start = chunk_end + 1 - self.overlap_frames
                # Ensure we don't go backwards or get stuck
                if chunk_start >= num_frames:
                    break
                # Prevent infinite loop on small chunks
                if chunk_end >= num_frames - 1:
                    break
            
            # Filter short-lived objects
            self._filter_short_lived_objects()
            
            # Generate output video if requested
            if self.generate_video:
                self._generate_output_video(num_frames)
            
            self.logger.info(f"Tracking complete! Found {len(self.tracked_objects)} objects")
            
            return self.tracked_objects
            
        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            self._clear_memory()
    
    def _filter_short_lived_objects(self):
        """Remove objects that appear in too few frames"""
        to_remove = []
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.num_frames < self.min_tracked_frames:
                to_remove.append(obj_id)
                self.logger.debug(f"Removing object {obj_id} with only {tracked_obj.num_frames} frames")
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        if to_remove:
            self.logger.info(f"Filtered out {len(to_remove)} short-lived objects (< {self.min_tracked_frames} frames)")
    
    def _generate_output_video(self, num_frames: int):
        """Generate output video with tracking visualization"""
        try:
            self.logger.info("Generating output video...")
            
            # Create video writer
            output_path = self.output_dir / "tracked_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, 
                self.video_info['fps'], 
                (self.video_info['width'], self.video_info['height'])
            )
            
            # Colors for different objects
            colors = {}
            for obj_id in self.tracked_objects.keys():
                colors[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Process each frame
            for frame_idx in range(num_frames):
                frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
                if not frame_path.exists():
                    continue
                
                frame = cv2.imread(str(frame_path))
                
                # Draw tracked objects
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if frame_idx in tracked_obj.frames:
                        box = tracked_obj.frames[frame_idx].astype(int)
                        color = colors[obj_id]
                        
                        # Draw box
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        
                        # Draw label
                        label = f"{tracked_obj.label} #{obj_id}"
                        if frame_idx in tracked_obj.confidences:
                            conf = tracked_obj.confidences[frame_idx]
                            label += f" ({conf:.2f})"
                        
                        cv2.putText(frame, label, (box[0], box[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                out.write(frame)
                
                # Clear memory periodically
                if frame_idx % 100 == 0:
                    self.logger.debug(f"Generated {frame_idx}/{num_frames} frames")
            
            out.release()
            self.logger.info(f"Output video saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate output video: {e}")


# Convenience wrapper to match existing API
class Sam2VideoTracker(ChunkedSam2VideoTracker):
    """Wrapper class for backward compatibility"""
    pass


def create_chunked_tracker(
    video_path: str,
    detection_model,
    **kwargs
) -> ChunkedSam2VideoTracker:
    """Create a chunked video tracker"""
    return ChunkedSam2VideoTracker(
        video_path=video_path,
        detection_model=detection_model,
        **kwargs
    )