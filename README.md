## RoBERTa Q&A with Wiki tools

This is a streamlit app which demonstrates the answer-retrieval capabilities of a fine-tuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.  [Click here to visit the 🤗 model card](https://huggingface.co/etweedy/roberta-base-squad-v2) to see more details about this model or [click here to visit a 🤗 space hosting the app](https://huggingface.co/spaces/etweedy/roberta-squad-v2) to try it out in the browser.

* The model is a fine-tuned version of [roberta-base for QA](https://huggingface.co/roberta-base)
* It was fine-tuned for context-based extractive question answering on the [SQuAD v2 dataset](https://huggingface.co/datasets/squad_v2), a dataset of English-language context-question-answer triples designed for extractive question answering training and benchmarking.
* Version 2 of SQuAD (Stanford Question Answering Dataset) contains the 100,000 examples from SQuAD Version 1.1, along with 50,000 additional "unanswerable" questions, i.e. questions whose answer cannot be found in the provided context.
* The original RoBERTa (Robustly Optimized BERT Pretraining Approach) model was introduced in [this paper](https://arxiv.org/abs/1907.11692) and [this repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)

This app contains three levels of question-answering:
1. A basic Q&A tool which allows the user to ask the model to search a user-provided context paragraph for the answer to a user-provided question.
2. A user-guided Wiki Q&A tool which allows the user to search for one or more Wikipedia pages and ask the model to search those pages for the answer to a user-provided question.
3. An automated Wiki Q&A tool which asks the model to perform retrieve its own Wikipedia pages in order to answer the user-provided question.
