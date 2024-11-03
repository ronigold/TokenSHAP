# TokenSHAP: Implementing the Paper with Monte Carlo Shapley Value Estimation

TokenSHAP is a Python library designed to implement the method described in the paper "TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation" (Goldshmidt & Horovicz, 2024). This package introduces a novel approach for interpreting large language models (LLMs) by estimating Shapley values for individual tokens, providing insights into how specific parts of the input contribute to the model’s decisions.

![Tokens Architecture](data/TokenSHAP_flow.png)

TokenSHAP offers a novel method for interpreting large language models (LLMs) using Monte Carlo Shapley value estimation. This Python library attributes importance to individual tokens within input prompts, enhancing our understanding of model decisions. By leveraging concepts from cooperative game theory adapted to the dynamic nature of natural language, TokenSHAP facilitates a deeper insight into how different parts of an input contribute to the model's response.

![Tokens Importance](data/plot.JPG)

## About TokenSHAP

The method introduces an efficient way to estimate the importance of tokens based on Shapley values, providing interpretable, quantitative measures of token importance. It addresses the combinatorial complexity of language inputs and demonstrates efficacy across various prompts and LLM architectures. TokenSHAP represents a significant advancement in making AI more transparent and trustworthy, particularly in critical applications such as healthcare diagnostics, legal analysis, and automated decision-making systems.

## Prerequisites

Before installing TokenSHAP, you need to have Ollama deployed and running. Ollama is required for TokenSHAP to interact with large language models.

To install and set up Ollama, please follow the instructions in the [Ollama GitHub repository](https://github.com/ollama/ollama).

## Installation

You can install TokenSHAP directly from PyPI using pip:

```bash
pip install tokenshap
```

Alternatively, to install from source:

```bash
git clone https://github.com/ronigold/TokenSHAP.git
cd TokenSHAP
pip install -r requirements.txt
```

## Usage

TokenSHAP is easy to use with any model that supports SHAP value computation for NLP. 
Here's a quick guide:

- Local Model Usage:

```python
# Import TokenSHAP
from token_shap import TokenSHAP

model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
model = LocalModel(model_name_or_path)
splitter = StringSplitter()
token_shap = TokenSHAP(model, splitter)

# Analyze token importance
prompt = "Why is the sky blue?"
df = token_shap.analyze(prompt, sampling_ratio=0.0, print_highlight_text=True)
```

- API Model Usage:

```python
# Import TokenSHAP
from token_shap import TokenSHAP

model_name = "llama3.2:3b"
api_url = "http://localhost:11434"

api_model = OllamaModel(model_name=model_name, api_url=api_url)
splitter = StringSplitter()
token_shap_api = TokenSHAP(api_model, splitter, debug=False)

# Analyze token importance
prompt = "Why is the sky blue?"
df = token_shap_api.analyze(prompt, sampling_ratio=0.0, print_highlight_text=True)
```

Results will include SHAP values for each token, indicating their contribution to the model's output.

For a more detailed example and usage guide, please refer to our [TokenSHAP Examples notebook](notebooks/TokenShap%20Examples.ipynb) in the repository.

## Key Features

- **Interpretability for LLMs:** Delivers a methodical approach to understanding how individual components of input affect LLM outputs.
- **Monte Carlo Shapley Estimation:** Utilizes a Monte Carlo approach to efficiently compute Shapley values for tokens, suitable for extensive texts and large models.
- **Versatile Application:** Applicable across various LLM architectures and prompt types, from factual questions to complex multi-sentence inputs.

## Contributing

We welcome contributions from the community, whether it's adding new features, improving documentation, or reporting bugs. Here's how you can contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourAmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/YourAmazingFeature`)
5. Open a pull request

## Support

For support, please email roni.goldshmidt@getnexar.com or miriam.horovicz@ni.com, or open an issue on our GitHub project page.

## License

TokenSHAP is distributed under the MIT License. See `LICENSE` file for more information.

## Citation

If you use TokenSHAP in your research, please cite our paper:

```
@article{goldshmidt2024tokenshap,
  title={TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation},
  author={Goldshmidt, Roni and Horovicz, Miriam},
  journal={arXiv preprint arXiv:2407.10114},
  year={2024}
}
```

You can find the full paper on arXiv: [https://arxiv.org/abs/2407.10114](https://arxiv.org/abs/2407.10114)

## Authors

- **Roni Goldshmidt**
- **Miriam Horovicz**
