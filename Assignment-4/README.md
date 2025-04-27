# Self-RAG for Enterprise AML

## Introduction
This repository contains code for implementing Self-Reflective Retrieval-Augmented Generation (Self-RAG) for Anti-Money Laundering (AML) applications in enterprise settings. The system enables financial professionals to query AML knowledge using natural language and receive accurate, well-grounded responses.

## Project Overview

The project follows this workflow:
1. Web scraping & Data Formatting
2. Retriever Design & Experiments (including fine-tuning)
3. Self-RAG Inference
4. Results Evaluation

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch torch==2.1.2
- FAISS
- HuggingFace Transformers
- VLLM
- Install the selfrag environment in https://github.com/AkariAsai/self-rag.git, search for enviroment.yml or requirements.txt to set up the environment.

### Installation

```bash
# Clone the repository
git clone https://github.com/peiqiliu777/NLP_Assignment-3.git
cd self-rag-main

# Create and activate a virtual environment
python -m venv selfrag
source selfrag/bin/activate  # On Windows: selfrag\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

# README

## Data Processing Pipeline

### 1. Extract Data
First, extract the raw FDIC data:
```bash
python get_data/get_data.py
```
Output: `fdic_extracted_data.json` - Raw extracted FDIC data

### 2. Clean Data
Next, clean the extracted data:
```bash
python get_data/data_cleaning.py
```
Input: `fdic_extracted_data.json` - Raw FDIC data
Output: `fdic_extracted_cleaned.json` - Cleaned FDIC data

### 3. Translate Data
Then, translate the cleaned data using strict parameters:
```bash
python get_data/translator.py
```
Input: `fdic_extracted_cleaned.json` - Cleaned FDIC data
Output: `fdic_extracted_translated_strict.json` - Translated FDIC data

### 4. Generate QA Dataset
Finally, generate the question-answer dataset based on topics:
```bash
python qa_generator/QA_topicbased_generate.py
```
Input: `fdic_extracted_translated_strict.json` - Translated FDIC data
Output: `full_qa_dataset.json` - Complete question-answer dataset


## Retriever Fine-tuning

### 1. Prepare Training Data

Prepare training data for retriever fine-tuning:

```bash
python prepare_retrieval_training.py --qa_file train_qa_pairs.json --doc_file aml_clean.json --output_file retriever_training_data.json
```

Input:
- `train_qa_pairs.json` - Question-Answer pairs with relevant documents
- `aml_clean.json` - Cleaned document chunks
Output: `retriever_training_data.json` - Formatted training data for contrastive learning

### 2. Fine-tune Retriever (If use baseline default retriever, ignore this step)

Fine-tune the Contriever model:

```bash
python fine_tune_retriever.py --model_name facebook/contriever --train_file retriever_training_data.json --output_dir fine_tuned_retriever
```

Input: `retriever_training_data.json` - Training data for contrastive learning
Output: `fine_tuned_retriever/` - Directory containing the fine-tuned model

### 3. Rebuild Index with Fine-tuned Model

Rebuild the retrieval index using the fine-tuned model:

```bash
python rebuild_index.py --model_path fine_tuned_retriever --input_file aml_clean.json --output_index aml_finetuned_index.faiss --output_meta aml_finetuned_meta.json
```

Input:
- `fine_tuned_retriever/` - Fine-tuned retriever model
- `aml_clean.json` - Cleaned document chunks
Output:
- `aml_finetuned_index.faiss` - FAISS index using fine-tuned embeddings
- `aml_finetuned_meta.json` - Metadata for the indexed documents

## Dataset Preparation for Self-RAG

Update the dataset with retrieved documents:

```bash
# For fine-tuned retriever
python update_dataset_with_retriever.py --model_path fine_tuned_retriever --index_path aml_finetuned_index.faiss --meta_path aml_finetuned_meta.json --input_files full_qa_dataset.json --output_file aml_self_rag_dataset.json --ndocs 5

# For default retriever
python update_dataset_with_retriever.py --model_name facebook/contriever --index_path aml_index.faiss --meta_path aml_meta.json --input_files full_qa_dataset.json --output_file aml_self_rag_dataset_default.json --ndocs 5
```

Input:
- `full_qa_dataset.json' complete dataset including 557 qa_pairs
- Fine-tuned or default retriever model and index
Output:
- `aml_self_rag_dataset.json` or `aml_self_rag_dataset_default.json` - Dataset with retrieved documents

## Self-RAG Inference

Run inference using the Self-RAG model:

```bash
# Using fine-tuned retriever results
python run_long_form_static.py \
  --model_name /path/to/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file aml_self_rag_dataset.json \
  --output_file finetuned_result.json --max_depth 7 --mode always_retrieve \
  --dtype float32

# Using default retriever results
python run_long_form_static.py \
  --model_name /path/to/selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file aml_self_rag_dataset_default.json \
  --output_file default_result.json --max_depth 7 --mode always_retrieve \
  --dtype float32
```

Input: `aml_self_rag_dataset.json` or `aml_self_rag_dataset_default.json`
Output: `finetuned_result.json` or `default_result.json` - Generated answers

## Evaluation

Evaluate the results using multiple metrics:

```bash
# Standard metrics evaluation, replace the input file for different methods' evaluations
python eval.py --f your_result.json --qa --mauve

# With LLM-based evaluation, replace the input file for different methods' evaluations
python eval_test.py --input /results/default_result.json --api_key YOUR_API_KEY

```



## Parameter Guide

### Retriever Fine-tuning Parameters
- `learning_rate`: 2e-5 (recommended)
- `batch_size`: 8-16 depending on GPU memory
- `num_epochs`: 3-5 for most datasets
- `temperature`: 0.05-0.1 for contrastive loss

### Self-RAG Inference Parameters
- `ndocs`: Number of documents to retrieve per question (4-5 recommended)
- `max_new_tokens`: Maximum number of tokens to generate (300-400 for detailed answers)
- `threshold`: Threshold for retrieval decision (0.2 recommended)
- `max_depth`: Maximum depth of the search tree (7 recommended for complex answers)
- `use_grounding`: Enable grounding score computation
- `use_utility`: Enable utility score computation
- `use_seqscore`: Enable sequence score computation
- `mode`: Retrieval mode (`adaptive_retrieval`, `always_retrieve`, or `no_retrieval`)

## Experimental Configurations

Here are some recommended configurations:

1. **Fine-tuned retriever with simple prompt**:
   ```
   --ndocs 4 --max_new_tokens 400 --threshold 0.2 --max_depth 7
   ```

2. **Fine-tuned retriever with detailed prompt**:
   ```
   --ndocs 5 --max_new_tokens 300 --threshold 0.2 --max_depth 7
   ```

3. **Default retriever baseline**:
   ```
   --ndocs 5 --max_new_tokens 300 --threshold 0.2 --max_depth 7
   ```

## Evaluation Metrics

The system is evaluated using multiple metrics:

- **QA-F1** and **QA-EM**: Measure answer correctness
- **STR-EM** and **STR-Hit**: Measure retrieval precision
- **ROUGE-L**: Measures overlap with reference answers
- **MAUVE**: Measures text naturalness and fluency
- **LLM-based metrics**: Factual correctness, information completeness, relevance, and retrieval utilization

## Troubleshooting

1. **Document ID Format Mismatch**: If you see all zeros in evaluation metrics, check for document ID format mismatches between retrieved documents and test data.

2. **GPU Memory Issues**: Reduce batch size or use gradient accumulation if encountering memory errors during fine-tuning.

3. **Retriever Quality**: If retrieval quality is poor, try increasing the number of training examples or adjusting the contrastive loss temperature.

## Citation

```
@article{asai2023selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yu and Sil, Avi and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2310.11511},
  year={2023}
}
```

## Contributors

Yanxuan Dong, Peiqi Liu - Carnegie Mellon University