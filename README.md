# 🧪 Retrieval-Augmented Test Generation: How Far Are We?

> A comprehensive replication package for evaluating RAG-based unit test generation across popular ML libraries

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/my-paper) -->

## 📋 Table of Contents

- [🔍 Overview](#-overview)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Dataset](#-dataset)
- [🛠️ Usage](#️-usage)
  - [1. Data Crawling](#1-data-crawling)
  - [2. RAG Database Creation](#2-rag-database-creation)
  - [3. Unit Test Generation](#3-unit-test-generation)
  - [4. Evaluation](#4-evaluation)
- [🔬 Analysis Scripts](#-analysis-scripts)
- [📈 Results](#-results)
- [🔧 Prompt Engineering Design](#-prompt-engineering-design)
- [🐛 Bug Reports](#-bug-reports)
- [📚 Citation](#-citation)

## 🔍 Overview

This research investigates the effectiveness of Retrieval-Augmented Generation (RAG) for automated unit test generation across five popular machine/deep learning libraries: **TensorFlow**, **PyTorch**, **Scikit-learn**, **JAX**, and **XGBoost**.

Our approach combines multiple RAG strategies:
- 📖 **API Documentation RAG**: Using official documentation
- 🔍 **GitHub Issues RAG**: Leveraging community-reported issues
- 💬 **StackOverflow RAG**: Utilizing Q&A discussions
- 🎯 **Combined RAG**: Integrating all sources

## 🏗️ Project Structure

```
api_guided_testgen/
├── 📁 crawling/                    # Data collection scripts
│   ├── crawl_from_apidoc.py       # API documentation crawler
│   ├── crawl_from_github_issues.py # GitHub issues crawler
│   ├── crawl_from_sos.py          # StackOverflow crawler
│   ├── crawl_github_repo.py       # GitHub repository crawler
│   ├── get_api_using_code.py      # API extraction from code
│   └── util.py                    # Utility functions
├── 📁 data/                       # Dataset and API lists
│   └── *_api_list.txt            # API lists per library
├── 📁 manual_analysis/            # Manual analysis artifacts (RQ4 & RQ5)
│   ├── README.md                  # Manual analysis documentation
│   ├── codebook.md                # Document classification framework
│   ├── coding_process.md          # Independent coding methodology
│   ├── initial_categories.md      # Initial category framework
│   ├── reported_bugs.csv          # Bug reports (RQ4)
│   ├── manual_data.csv            # Manual evaluation data (RQ5)
│   └── detailed_analysis.csv      # Classification results (RQ5)
├── 📁 prompt_engineering/         # Prompt design and evolution
│   ├── README.md                  # Prompt engineering documentation
│   ├── version1.py                # Initial prompt version
│   ├── version2.py                # Improved structure version
│   ├── version3.py                # Zero-shot baseline version
│   └── version4.py                # RAG-enhanced version
├── 📁 out/                        # Generated test outputs
├── 🐍 api_rag.py                  # Main test generation script
├── 🐍 generate_ragdoc.py          # RAG database creation
├── 🐍 evaluate.py                 # Evaluation script
├── 🐍 coverage.py                 # Coverage analysis
├── 🐍 util.py                     # Utility functions
├── 🐍 posthoc_test.py             # Post-hoc analysis
├── 🔧 run_whole_pipeline.sh       # Complete pipeline automation
└── 🔧 make_new_apiragdoc.sh       # RAG document generation script
```

## ⚙️ Installation

### Prerequisites
- 🐍 Python 3.8+
- 🔑 API keys for data collection (optional)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `openai` - OpenAI API integration
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Text embeddings
- `fireworks-ai` - Alternative LLM provider
- `tensorflow` - TensorFlow library testing
- `torch` - PyTorch library testing
- `scikit-learn` - Scikit-learn library testing
- `jax` - JAX library testing
- `xgboost` - XGBoost library testing
- `tqdm` - Progress bars
- `python-dotenv` - Environment variable management

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
GITHUB_TOKEN=your_github_token_here
STACKEXCHANGE_API_KEY=your_stackexchange_key_here
```

## 🚀 Quick Start

### Run the Complete Pipeline

```bash
# Generate and evaluate tests for TensorFlow using API RAG
./run_whole_pipeline.sh tf api_rag_all output_dir gpt-4o
```

### Step-by-Step Execution

```bash
# 1. Create RAG database
python generate_ragdoc.py tf api_rag_all

# 2. Generate tests
python api_rag.py tf api_rag_all ./output gpt-4o

# 3. Evaluate results
python evaluate.py tf api_rag_all ./output
```

### Data Collection

Our dataset includes:
- 📖 **API Documentation**: Official docs with examples
- 🔍 **GitHub Issues**: Community-reported problems and solutions
- 💬 **StackOverflow**: Q&A discussions and code snippets

API lists are provided in `data/{library}_api_list.txt` files.

## 🛠️ Usage

### 1. Data Crawling

Collect documentation and community data:

```bash
# Crawl API documentation
python crawling/crawl_from_apidoc.py [tf|torch|sklearn|xgb|jax]

# Crawl GitHub issues and comments
python crawling/crawl_from_github_issues.py [tf|torch|sklearn|xgb|jax] [comments|issues]

# Crawl StackOverflow Q&As
python crawling/crawl_from_sos.py [tf|torch|sklearn|xgb|jax]
```

**Note**: 🔑 You'll need API credentials for GitHub and StackOverflow crawling.

### 2. RAG Database Creation

Build vector databases for retrieval:

```bash
python generate_ragdoc.py [LIBRARY] [RAG_TYPE]
```

**Libraries**: `tf`, `torch`, `sklearn`, `xgb`, `jax`

**RAG Types**:
- `basic_rag_all` - Combined basic RAG
- `basic_rag_apidoc` - API documentation only
- `basic_rag_issues` - GitHub issues only
- `basic_rag_sos` - StackOverflow only
- `api_rag_all` - Combined API-guided RAG ⭐
- `api_rag_apidoc` - API-guided documentation
- `api_rag_issues` - API-guided issues
- `api_rag_sos` - API-guided StackOverflow

**Database Storage**:
- 📁 `./docs_db/` - Basic RAG databases ([Download Link](https://drive.google.com/file/d/1ZPUkVsXPVQhTloCPDsl_sbIxCEXBF010/view?usp=sharing))
- 📁 `./api_docs_db/` - API RAG databases ([Download Link](https://drive.google.com/file/d/1ZMHxYDiumEUk_eby7RKQlodLEAs7TIwo/view?usp=sharing))
### 3. Unit Test Generation

Generate unit tests using different approaches:

```bash
python api_rag.py [LIBRARY] [APPROACH] [OUTPUT_DIR] [MODEL]
```

**Parameters**:
- **LIBRARY**: Target ML library
- **APPROACH**: RAG strategy or `zero_shot`
- **OUTPUT_DIR**: Where to save generated tests
- **MODEL**: `gpt-3.5-turbo`, `gpt-4o`, `mistral`, `llama`

**Example**:
```bash
python api_rag.py tf api_rag_all ./output gpt-4o
```

### 4. Evaluation

Evaluate the generated tests:

```bash
# Basic evaluation (parse, exec, pass rates)
python evaluate.py [LIBRARY] [APPROACH] [OUTPUT_DIR]

# Coverage analysis
python coverage.py [LIBRARY] [APPROACH] [OUTPUT_DIR]

# Detailed metrics
python util.py [LIBRARY] [APPROACH] [OUTPUT_DIR]
```

**Output Metrics**:
- 📊 **Parse Rate**: Syntactically correct tests
- ⚡ **Execution Rate**: Tests that run without errors
- ✅ **Pass Rate**: Tests that pass assertions
- 📈 **Line Coverage**: Code coverage achieved
- 🔢 **Token Usage**: Input/output token statistics

### 5. Replicate Output
In order to replicate the experiment.
We are also providing the generated test cases and their execution results `out/`.

[Download Link](https://drive.google.com/file/d/1S-HxiFJ1-Pq3mYEO_N9DlmuEpk0WBEFG/view?usp=sharing)
## 🔧 Prompt Engineering Design

We developed a straightforward two-step approach to prompt engineering for automated unit test generation across ML/DL libraries.

### Approach Overview

Our method evolved through four iterations to reach the final prompts:

1. **Version 1**: Initial basic prompt with minimal structure
2. **Version 2**: Improved structure with unittest framework
3. **Version 3**: Final zero-shot baseline with coverage focus (used in research)
4. **Version 4**: RAG-enhanced with retrieved documentation (used in research)

### Key Design Principles

- **Simplicity**: Keep prompts concise and focused on core requirements
- **Executable Focus**: Emphasize generating runnable test code with proper structure
- **Context Integration**: Use retrieved information to improve test quality and coverage
- **Practical Testing**: Generate tests that work in real-world scenarios

### Detailed Documentation

For complete prompt details and example outputs, see the **[`prompt_engineering/`](./prompt_engineering/)** directory.

## 🐛 Bug Reports and Manual Analysis

Our research includes comprehensive analysis of both reported bugs and manual evaluation on RAG documents influence on test generation/

### Bug Reports (RQ4)
Our generated tests discovered **28 bugs** and out of them **24 are unique new bugs**.

### Manual Analysis (RQ5)
We conducted detailed manual analysis of retrieved documents, generated tests, and the unique code they covered. This analysis categorized document types based on their influence on test generation, including bug-inducing patterns, API usage sequences, unique/complex input parameters, exception handling, integration patterns, performance considerations, and standard use cases.

**📁 For detailed methodology, data, and results, see the [`manual_analysis/`](./manual_analysis/) directory.**

## 📚 Citation

If you use this work, please cite (arXiv version for now): 

```bibtex
@article{shin2024retrieval,
  title={Retrieval-augmented test generation: How far are we?},
  author={Shin, Jiho and Harzevili, Nima Shiri and Aleithan, Reem and Hemmati, Hadi and Wang, Song},
  journal={arXiv preprint arXiv:2409.12682},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. 🍴 Fork the repository
2. 🌟 Create a feature branch
3. 📝 Make your changes
4. 🧪 Add tests
5. 📬 Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Hugging Face for embeddings
- The open-source ML community
- GitHub and StackOverflow communities

---

<div align="center">
  <strong>🚀 Happy Testing! 🚀</strong>
</div>