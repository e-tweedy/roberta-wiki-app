import torch
import streamlit as st
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)
import spacy
# import pandas as pd
from lib.utils import ContextRetriever


#### TO DO:######
# build out functions for:
#     * formatting input into document retrieval query (spaCy)
#     * document retrieval based on query (wikipedia library)
#     * document postprocessing into passages
#     * ranking passage based on BM25 scores for query (rank_bm25)
#     * feeding passages into RoBERTa an reporting answer(s) and passages as evidence
# decide what to do with examples

### CAN REMOVE:#####
    # * context collection
    # *

########################
### Helper functions ###
########################

# Build trainer using model and tokenizer from Hugging Face repo
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Load model and tokenizer from 🤗 repo
    and build pipeline
    Parameters: None
    -----------
    Returns:
    --------
    qa_pipeline : transformers.QuestionAnsweringPipeline
        The question answering pipeline object
    """
    repo_id = 'etweedy/roberta-base-squad-v2'
    qa_pipeline = pipeline(
        task = 'question-answering',
        model=repo_id,
        tokenizer=repo_id,
        handle_impossible_answer = True
    )
    return qa_pipeline

@st.cache_resource(show_spinner=False)
def get_spacy():
    """
    Load spaCy model for processing query
    Parameters: None
    -----------
    Returns:
    --------
    nlp : spaCy.Pipe
        Portion of 'en_core_web_sm' model pipeline
        only containing tokenizer and part-of-speech
        tagger
    """
    nlp = spacy.load(
        'en_core_web_sm',
        disable = ['ner','parser','textcat']
    )
    return nlp

def generate_query(nlp,text):
    """
    Process text into a search query,
    only retaining nouns, proper nouns,
    numerals, verbs, and adjectives
    Parameters:
    -----------
    nlp : spacy.Pipe
        spaCy pipeline for processing search query
    text : str
        The input text to be processed
    Returns:
    --------
    query : str
        The condensed search query
    """
    tokens = nlp(text)
    keep = {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}
    query = ' '.join(token.text for token in tokens \
                         if token.pos_ in keep)
    return query

def fill_in_example(i):
    """
    Function for context-question example button click
    """
    st.session_state['response'] = ''
    st.session_state['question'] = ex_q[i]

def clear_boxes():
    """
    Function for field clear button click
    """
    st.session_state['response'] = ''
    st.session_state['question'] = ''

# def get_examples():
#     """
#     Retrieve pre-made examples from a .csv file
#     Parameters: None
#     -----------
#     Returns:
#     --------
#     questions, contexts : list, list
#         Lists of examples of corresponding question-context pairs
        
#     """
#     examples = pd.read_csv('examples.csv')
#     questions = list(examples['question'])
#     return questions

#############
### Setup ###
#############
    
# Set mps or cuda device if available
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Initialize session state variables
if 'response' not in st.session_state:
    st.session_state['response'] = ''
if 'question' not in st.session_state:
    st.session_state['question'] = ''

# Retrieve trained RoBERTa pipeline for Q&A
# and spaCy pipeline for processing search query
with st.spinner('Loading the model...'):
    qa_pipeline = get_pipeline()
    nlp = get_spacy()

# # Grab example question-context pairs from csv file
# ex_q, ex_c = get_examples()

###################
### App content ###
###################

# Intro text
st.header('RoBERTa answer retieval')
st.markdown('''
This app demonstrates the answer-retrieval capabilities of a fine-tuned RoBERTa (Robustly optimized Bidirectional Encoder Representations from Transformers) model.

Please type in a question and click submit.  When you do, a few things will happen:
1. A Wikipedia search will be performed based on your question
2. Candidate passages will be ranked based on a similarity score as compared to your question
3. RoBERTa will search the best candidate passages to find the answer to your question

If the model cannot find the answer to your question, it will tell you so.
''')
with st.expander('Click to read more about the model...'):
    st.markdown('''
* [Click here](https://huggingface.co/etweedy/roberta-base-squad-v2) to visit the Hugging Face model card for this fine-tuned model.
* To create this model, the [RoBERTa base model](https://huggingface.co/roberta-base) was fine-tuned on Version 2 of [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/squad_v2), a dataset of context-question-answer triples.
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
# st.markdown('''
# Please type or paste a context paragraph and question you'd like to ask about it.  The model will attempt to answer the question based on the context you provided.  If the model cannot find the answer in the context, it will tell you so - the model is also trained to recognize when the context doesn't provide the answer.

# Your results will appear below the question field when the model is finished running.

# Alternatively, you can try an example by clicking one of the buttons below:
# ''')

# Generate containers in order
# example_container = st.container()
input_container = st.container()
button_container = st.container()
response_container = st.container()

###########################
### Populate containers ###
###########################

# Populate example button container
# with example_container:
#     ex_cols = st.columns(len(ex_q)+1)
#     for i in range(len(ex_q)):
#         with ex_cols[i]:
#             st.button(
#                 label = f'Try example {i+1}',
#                 key = f'ex_button_{i+1}',
#                 on_click = fill_in_example,
#                 args=(i,),
#             )
#     with ex_cols[-1]:
#         st.button(
#             label = "Clear all fields",
#             key = "clear_button",
#             on_click = clear_boxes,
#         )

# Populate user input container
with input_container:
    with st.form(key='input_form',clear_on_submit=False):
        # Question input field
        question = st.text_input(
            label='Question',
            value=st.session_state['question'],
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        # Form submit button
        query_submitted = st.form_submit_button("Submit")
        if query_submitted:
            # update question, context in session state
            st.session_state['question'] = question
            with st.spinner('Retrieving documentation...'):
                query = generate_query(nlp,question)
                retriever = ContextRetriever()
                retriever.get_pageids(query)
                retriever.get_pages()
                retriever.get_paragraphs()
                retriever.rank_paragraphs(question)
            with st.spinner('Generating response...'):
                # Loop through best_paragraph contexts
                # looking for answer in each
                best_answer = ""
                for context in retriever.best_paragraphs:
                    input = {
                        'context':context,
                        'question':st.session_state['question'],
                    }
                    # Pass to QA pipeline
                    response = qa_pipeline(**input)
                    if response['answer']!='':
                        best_answer = response['answer']
                        best_context = context
                        break
            # Update response in session state
            if best_answer == "":
                st.session_state['response'] = "I cannot find the answer to your question."
            else:
                st.session_state['response'] = f"""
                    My answer is: {best_answer}
                    
                    ...and here's where I found it:
                    
                    {best_context}
                """

# Button for clearing the form
with button_container:
    st.button(
            label = "Clear all fields",
            key = "clear_button",
            on_click = clear_boxes,
        )

# Display response
with response_container:
    st.write(st.session_state['response'])
            
