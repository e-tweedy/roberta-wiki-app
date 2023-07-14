import torch
import streamlit as st
# from datasets import Dataset
# from torch.utils.data import DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoModelForQuestionAnswering,
#     pipeline,
# )
# from lib.utils import (
#     clear_boxes,
#     get_examples,
#     fill_in_example,
# )
########################
### Helper functions ###
########################

# # Build trainer using model and tokenizer from Hugging Face repo
# @st.cache_resource(show_spinner=False)
# def get_pipeline():
#     """
#     Load model and tokenizer from 🤗 repo
#     and build pipeline
#     Parameters: None
#     -----------
#     Returns:
#     --------
#     qa_pipeline : transformers.QuestionAnsweringPipeline
#         The question answering pipeline object
#     """
#     repo_id = 'etweedy/roberta-base-squad-v2'
#     qa_pipeline = pipeline(
#         task = 'question-answering',
#         model=repo_id,
#         tokenizer=repo_id,
#         handle_impossible_answer = True
#     )
#     return qa_pipeline

#############
### Setup ###
#############
    
# # Set mps or cuda device if available
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# # Initialize session state variables
# for page in ['basic','semi','auto']:
#     if page not in st.session_state:
#         st.session_state[page] = {}
#     for field in ['question','response']:
#         if field not in st.session_state[page]:
#             st.session_state[page][field]=''
# if 'query' not in st.session_state['semi']:
#     st.session_state['semi']['query'] = ''
# if 'context' not in st.session_state['basic']:
#     st.session_state['basic']['context'] = ''

# # Retrieve stored model
# with st.spinner('Loading the model...'):
#     qa_pipeline = get_pipeline()

# # Retrieve example questions and contexts
# ex_questions, ex_contexts = get_examples()

###################
### App content ###
###################

# Intro text
st.header('RoBERTa Q&A model')
st.markdown('''
This app demonstrates the answer-retrieval capabilities of a fine-tuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.
''')
with st.expander('Click to read more about the model...'):
    st.markdown('''
* [Click here](https://huggingface.co/etweedy/roberta-base-squad-v2) to visit the Hugging Face model card for this fine-tuned model.
* To create this model, I fine-tuned the [RoBERTa base model](https://huggingface.co/roberta-base) Version 2 of [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/squad_v2), a dataset of context-question-answer triples.
* The objective of the model is "extractive question answering", the task of retrieving the answer to the question from a given context text corpus.
* SQuAD Version 2 incorporates the 100,000 samples from Version 1.1, along with 50,000 'unanswerable' questions, i.e. samples in the question cannot be answered using the context given.
* The original base RoBERTa model was introduced in [this paper](https://arxiv.org/abs/1907.11692) and [this repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).  Here's a citation for that base model:
```bibtex
@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
''')
st.markdown('''
Use the menu on the left side to navigate between different app components:
1. A basic Q&A tool which allows the user to ask the model to search a user-provided context paragraph for the answer to a user-provided question.
2. A user-guided Wiki Q&A tool which allows the user to search for one or more Wikipedia articles and ask the model to search those articles for the answer to a user-provided question.
3. An automated Wiki Q&A tool which asks the model to perform retrieve its own Wikipedia articles in order to answer the user-provided question.
''')