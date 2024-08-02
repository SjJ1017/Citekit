# Citekit


<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a><a href="https://github.com/SjJ1017/Citekit/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>



## Overview

**Citekit** is an open-source, extensible toolkit designed to facilitate the implementation and evaluation of citation generation methods for Large Language Models (LLMs). It offers a modular framework to standardize citation tasks, enabling reproducible and comparable research while fostering the development of new approaches to improve citation quality.


## Features

+ **Modular Design**: Citekit is composed of four main modules: **INPUT**, **GENERATION MODULE**, **ENHANCING MODULE**, and **EVALUATOR**. These modules can be combined to construct pipelines for various citation generation tasks.
+ **Extensibility**: Easily extend Citekit by adding new components or modifying existing ones. The toolkit supports different LLM frameworks, including Hugging Face and API-based models like OpenAI.
+ **Comprehensive Evaluation**: Citekit includes predefined metrics for evaluating both answer quality and citation quality, with support for custom metrics.
+ **Predefined Recipes**: The toolkit provides 11 baseline recipes derived from state-of-the-art research, allowing users to quickly implement and compare different citation generation methods.

## Installation

To install Citekit, clone the repository from GitHub:

```bash
git clone https://github.com/SjJ1017/Citekit.git
cd Citekit
pip install -r requirements.txt
```

## Usage

### Run a Citation Generation Pipeline

To realize an existing pipeline, for example:

```bash
export PYTHONPATH="$PWD"
python methods/ALCE_Vani_Summ_VTG.py --mode text --pr --rouge --qa
```

Some files contain multiple methods. Use --mode to specify the desired method. For any pre-defined metrics, use --metric to enable it.

### Constructing a Citation Generation Pipeline
To construct a pipeline, follow the steps in the demonstration.ipynb file or our video on [Youtube](https://youtu.be/KaNICbbmCn0)

## Contributing

We welcome contributions to improve Citekit. Please submit pull requests or open issues on the [GitHub repository](https://github.com/SjJ1017/Citekit).

## License

This project is licensed under the MIT License - see the LICENSE file for details.


------
