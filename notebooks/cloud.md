# TokenSHAP Notebooks

## Overview
Example notebooks demonstrating TokenSHAP and PixelSHAP usage with various models and configurations.

## TokenSHAP Examples.ipynb

### Purpose
Demonstrates text token importance analysis using different models and vectorizers.

### Key Examples

**1. Basic TokenSHAP with OpenAI**
- GPT-4 model integration
- TF-IDF vectorization
- Word-level splitting
- Importance visualization

**2. Advanced Tokenization**
- HuggingFace tokenizer usage
- Subword analysis
- Custom split patterns
- Multi-line visualizations

**3. Local Model Analysis**
- HuggingFace transformers
- Sentence embeddings
- GPU acceleration
- Batch processing

**4. Comparative Analysis**
- Multiple model comparison
- Different vectorizer impacts
- Sampling ratio effects
- Performance benchmarking

### Visualization Examples
- Colored text output
- Heatmap generation
- Background highlighting
- Interactive plots

## PixelSHAP Examples.ipynb

### Purpose
Demonstrates object importance analysis in images using vision models.

### Key Examples

**1. Basic PixelSHAP Setup**
- YOLO object detection
- SAM2 segmentation
- Vision model integration
- Simple analysis pipeline

**2. Object Importance Analysis**
- Multi-object scenes
- Importance ranking
- Mask extraction
- Shapley value interpretation

**3. Advanced Visualizations**
- Heatmap overlays
- Side-by-side comparisons
- Transparency controls
- Label annotations

**4. Custom Manipulators**
- Different hiding strategies
- Blur effects
- Color filling
- Preservation masks

### Image Processing
- Segmentation visualization
- Bounding box display
- Mask combination
- Temporary file handling

## Common Patterns

### Model Initialization
```python
# OpenAI Vision
model = OpenAIModel(
    model_name="gpt-4-vision-preview",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Local Vision Model
model = LocalModel(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_type="vision",
    device="cuda"
)
```

### Vectorizer Setup
```python
# Simple TF-IDF
vectorizer = TfidfTextVectorizer()

# Semantic embeddings
vectorizer = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    device="cuda"
)
```

### Analysis Pipeline
```python
# TokenSHAP
results = token_shap.analyze(
    prompt=text,
    sampling_ratio=0.3,
    max_combinations=1000
)

# PixelSHAP
results, shapley_values = pixel_shap.analyze(
    image_path="image.jpg",
    prompt="Describe this image",
    sampling_ratio=0.5
)
```

## Utility Functions

### Results Processing
- DataFrame manipulation
- Shapley value sorting
- Similarity analysis
- Statistical summaries

### Visualization Helpers
- Color mapping functions
- Layout configuration
- Export utilities
- Interactive controls

## Data Files

### Example Images
- `nexar_1.png`: Traffic scene for object detection
- Sample outputs in `example_temp/`
- Segmentation masks cached

### Model Weights
- `yolov8x.pt`: YOLO detection model
- Auto-downloaded SAM2 weights
- Cached embeddings models

## Performance Considerations

### Sampling Strategies
- Start with ratio=0.1 for exploration
- Increase to 0.5 for detailed analysis
- Use max_combinations for time limits
- Essential combinations always included

### Memory Management
- Clear temporary files after analysis
- Batch process large datasets
- Monitor GPU memory usage
- Use smaller models for testing

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow analysis**: Lower sampling_ratio
3. **Poor segmentation**: Adjust detection thresholds
4. **API rate limits**: Add delays or use local models

### Debug Techniques
- Enable debug=True for detailed logs
- Visualize intermediate results
- Check model responses manually
- Verify segmentation masks

## Extensions

### Custom Analysis
- Modify combination generation
- Implement new vectorizers
- Add visualization styles
- Export to different formats

### Integration Examples
- Streamlit apps
- Gradio interfaces
- API endpoints
- Batch processing scripts