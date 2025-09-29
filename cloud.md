# TokenSHAP & PixelSHAP

## Overview
TokenSHAP is a Python library for interpreting large language models and vision models using Monte Carlo Shapley Value estimation. It provides tools to analyze token importance in text prompts (TokenSHAP) and object importance in images (PixelSHAP).

## Key Components

### Core Architecture
- **BaseSHAP**: Abstract base class providing Shapley value calculation framework
- **TokenSHAP**: Text token importance analysis for LLMs
- **PixelSHAP**: Object importance analysis for vision models using segmentation

### Model Support
- **OpenAI API**: GPT models via OpenAI client
- **Ollama**: Local model deployment support
- **HuggingFace**: Local transformers models
- **Vision Models**: Multimodal LLMs with image understanding

### Vectorization Methods
- **TF-IDF**: Basic text similarity measurement
- **HuggingFace Embeddings**: Sentence transformers for semantic similarity
- **OpenAI Embeddings**: API-based text embeddings

## Installation
```bash
pip install -r requirements.txt
```

## Core Dependencies
- numpy, pandas: Data processing
- matplotlib: Visualization
- scikit-learn: TF-IDF vectorization
- transformers: HuggingFace models
- sentence-transformers: Text embeddings
- opencv-python: Image processing (for PixelSHAP)
- ultralytics: YOLO object detection (optional)

## Quick Start

### TokenSHAP Example
```python
from token_shap import TokenSHAP
from token_shap.base import OpenAIModel, TfidfTextVectorizer, StringSplitter

# Initialize model and vectorizer
model = OpenAIModel(model_name="gpt-4", api_key="YOUR_KEY")
vectorizer = TfidfTextVectorizer()
splitter = StringSplitter(split_pattern=' ')

# Create TokenSHAP instance
token_shap = TokenSHAP(model=model, splitter=splitter, vectorizer=vectorizer)

# Analyze prompt
prompt = "Explain quantum computing in simple terms"
results = token_shap.analyze(prompt, sampling_ratio=0.5)

# Visualize
token_shap.plot_colored_text()
```

### PixelSHAP Example
```python
from token_shap import PixelSHAP
from token_shap.image_utils import YOLODetectionModel, Sam2Adapter
from token_shap.base import OpenAIModel, TfidfTextVectorizer

# Initialize components
model = OpenAIModel(model_name="gpt-4-vision", api_key="YOUR_KEY")
detector = YOLODetectionModel(model_path="yolov8x.pt")
segmenter = Sam2Adapter(sam2_model_id="facebook/sam2.1-hiera-large")
vectorizer = TfidfTextVectorizer()

# Create PixelSHAP instance
pixel_shap = PixelSHAP(
    model=model,
    segmentation_model=segmenter,
    manipulator=SegmentationBased(),
    vectorizer=vectorizer
)

# Analyze image
results, shapley_values = pixel_shap.analyze(
    image_path="image.jpg",
    prompt="What objects are in this image?",
    sampling_ratio=0.5
)

# Visualize importance
pixel_shap.visualize(heatmap_style=True, show_labels=True)
```

## Key Features

### TokenSHAP
- **Monte Carlo Sampling**: Efficient approximation of Shapley values for large prompts
- **Flexible Splitting**: Custom tokenization strategies (word, subword, character)
- **Multiple Visualizations**: Colored text, heatmaps, background highlighting
- **Sampling Control**: Adjust sampling_ratio for speed vs accuracy tradeoff

### PixelSHAP
- **Object Segmentation**: Automatic object detection and segmentation
- **Importance Heatmaps**: Visual overlays showing object importance
- **Mask Extraction**: Access to segmentation masks with Shapley values
- **Multiple Manipulators**: Different strategies for hiding objects (blur, fill, etc.)

## Algorithm Details

### Shapley Value Calculation
1. **Baseline Generation**: Get model response with full input
2. **Combination Sampling**: Generate subsets of tokens/objects
3. **Response Collection**: Get model outputs for each combination
4. **Similarity Measurement**: Compare responses to baseline using vectorization
5. **Shapley Computation**: Calculate marginal contributions
6. **Normalization**: Scale values to [0,1] range

### Sampling Strategy
- **Essential Combinations**: Always includes leave-one-out combinations
- **Random Sampling**: Additional combinations based on sampling_ratio
- **Max Combinations**: Optional limit for computational efficiency

## Configuration

### Debug Mode
Enable detailed logging:
```python
token_shap = TokenSHAP(model=model, splitter=splitter, vectorizer=vectorizer, debug=True)
```

### Sampling Parameters
- `sampling_ratio`: 0.0-1.0, controls number of combinations (0=essential only)
- `max_combinations`: Hard limit on total combinations
- Higher values = more accurate but slower

### Visualization Options
- `heatmap_style`: Color intensity based on importance
- `show_labels`: Display object/token labels
- `overlay_original`: Blend visualization with original image
- `opacity` controls: Fine-tune transparency levels

## Output Formats

### Results DataFrame
- **Content**: Token/object combination used
- **Response**: Model output for combination
- **Similarity**: Similarity score to baseline
- **Indexes**: Indices of included items

### Shapley Values Dictionary
- Keys: Token/object identifiers
- Values: Normalized importance scores (0-1)

### Saved Outputs
- `results.csv`: Full combination results
- `shapley_values.json`: Importance scores
- `metadata.json`: Analysis configuration

## Performance Tips
1. Start with low sampling_ratio (0.1-0.3) for initial exploration
2. Use max_combinations to cap computation time
3. Choose appropriate vectorizer for your use case
4. Enable debug mode to monitor progress
5. For images, limit object detection confidence threshold

## Error Handling
- Validates model responses before processing
- Handles empty detections gracefully
- Provides debug output for troubleshooting
- Automatic cleanup of temporary files

## Extension Points
- Custom model wrappers via ModelBase
- Custom vectorizers via TextVectorizer
- Custom splitters for TokenSHAP
- Custom manipulators for PixelSHAP
- Custom segmentation models