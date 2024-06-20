
# TokenSHAP

TokenSHAP is a Python library designed to enhance the interpretability of large language models by applying SHAP values to individual tokens within prompts. This library helps to understand the importance and influence of each token in model decisions, providing clearer insights into model behavior.

## Installation

```bash
git clone https://github.com/ronigold/TokenSHAP.git
cd TokenSHAP
pip install -r requirements.txt
```

## Usage

To use TokenSHAP, follow these simple steps:

```python
from token_shap import TokenSHAP

# Initialize TokenSHAP with your model
model_path = "path_to_your_model"
tokenizer_path = "path_to_your_tokenizer"
tshap = TokenSHAP(model_path, tokenizer_path)

# Analyze token importance
prompt = "Your prompt here"
analysis = tshap.analyze(prompt)
print(analysis)
```
![Tokens Importance](plot.png)

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Support

For support, email roni.goldshmidt@getnexar.com or open an issue on the GitHub project page.
