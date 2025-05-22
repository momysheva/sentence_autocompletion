# ✍️ Fine-Tuning Transformers for Literary Text Completion

This project explores fine-tuning large language models (LLMs) on literary texts using **parameter-efficient training** techniques, specifically **LoRA (Low-Rank Adaptation)**. We evaluate the models' ability to complete sentences in a stylistically coherent way using data derived from the **Project Gutenberg corpus**.

## 🎯 Objectives

- Fine-tune transformer models on English literary texts for sentence autocompletion.
- Evaluate and compare LoRA vs. full fine-tuning in low-resource environments.
- Use perplexity and BERTScore for quantitative evaluation, plus qualitative analysis.

## 📚 Dataset

- Source: [Project Gutenberg](https://www.gutenberg.org/)
- Preprocessing pipeline:
  - Removal of headers, footers, metadata, and non-linguistic elements.
  - Sentence tokenization using NLTK.
  - Filtering based on length and structure.
- Final Stats:
  - ~140,000 cleaned sentences
  - Train/Test split: 136,750 / 3,507

## 🤖 Models Used

All models were loaded from HuggingFace:
- GPT-2 (137M)
- GPT-Neo (125M)
- DistilGPT-2 (88.2M)
- DeepSeek-R1-Distill (1.7B)

### 🧪 Fine-Tuning Approaches

- **LoRA** (via `peft` library):
  - Rank: 8, Alpha: 32, Dropout: 0.1
  - Applied to attention and FFN layers
- **Full Fine-Tuning** (for comparison on selected models)

### 🔧 Training Setup

- Environment: Google Colab / Kaggle GPUs
- Objective: Causal Language Modeling
- Epochs: 5
- Batch Size: 16
- Learning Rate: 5e-4 (or 2e-4 for DeepSeek)

## 📊 Evaluation Metrics

- **Perplexity**: Lower = better sentence fluency
- **BERTScore**: Measures semantic similarity (Precision, Recall, F1)

## 📈 Results Summary

| Model                  | Finetuning Type | PPL ↓  | BERT F1 ↑ |
|------------------------|-----------------|--------|-----------|
| GPT-Neo                | Pretrained      | 101.84 | 0.43      |
|                        | LoRA            | 42.28  | 0.46      |
|                        | Full            | 13.70  | 0.59      |
| GPT-2                  | Pretrained      | 92.27  | 0.49      |
|                        | LoRA            | 41.94  | 0.50      |
|                        | Full            | 13.44  | 0.53      |
| DistilGPT-2            | Pretrained      | 155.60 | 0.49      |
|                        | LoRA            | 49.21  | 0.50      |
| DeepSeek-R1-Distill    | Pretrained      | 516.89 | 0.45      |
|                        | LoRA            | 55.23  | 0.51      |

## ✍️ Qualitative Results

Fine-tuned models demonstrated improved stylistic coherence, fluency, and semantic relevance in generated completions. Full fine-tuning achieved the best results but also showed signs of overfitting on rare words.

## 📎 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Project Gutenberg Dataset](https://huggingface.co/datasets/manu/project_gutenberg)
- [Anyscale Blog on LoRA](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)
