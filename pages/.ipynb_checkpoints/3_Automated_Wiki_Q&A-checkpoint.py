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
from lib.utils import (
    ContextRetriever,
    get_examples,
    generate_query,
)

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

def clear_boxes():
    """
    Clears the response and question fields
    """
    for field in ['question','response']:
        st.session_state['auto'][field]=''

def fill_in_example(i):
    """
    Fills in the chosen example question
    """
    st.session_state['auto']['question'] = ex_questions[i]

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
if 'auto' not in st.session_state:
    st.session_state['auto'] = {}
for field in ['question','response']:
    if field not in st.session_state['auto']:
        st.session_state['auto'][field] = ''

# Retrieve trained RoBERTa pipeline for Q&A
# and spaCy pipeline for processing search query
with st.spinner('Loading the model...'):
    qa_pipeline = get_pipeline()
    nlp = get_spacy()

# Retrieve example questions
_, ex_questions, _ = get_examples()
ex_questions = [q[0] for q in ex_questions]

###################
### App content ###
###################

# # Intro text
st.header('Automated Wiki Q&A')
st.markdown('''
This component attempts to automate the Wiki-assisted extractive question-answering task.  A Wikipedia search will be performed based on your question, and a list of relevant paragraphs will be passed to the RoBERTa model so it can attempt to find an answer.
''')
with st.expander("Click here to find out what's happening behind the scenes..."):
    st.markdown('''
    When you submit a question, the following steps are performed:
    1. Your question is condensed into a search query which just retains nouns, verbs, numerals, and adjectives, where part-of-speech tagging is done using the [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm) pipeline in the [spaCy library](https://spacy.io/).
    2. A Wikipedia search is performed using this query, resulting in several articles.  The articles from the top 3 search results are collected and split into paragraphs.  Wikipedia queries and article collection use the [wikipedia library](https://pypi.org/project/wikipedia/), a wrapper for the [MediaWiki API](https://www.mediawiki.org/wiki/API).
    4. The paragraphs are ranked in descending order of relevance to the query, using the [Okapi BM25 score](https://en.wikipedia.org/wiki/Okapi_BM25) as implemented in the [rank_bm25 library](https://github.com/dorianbrown/rank_bm25).
    5. The ten most relevant paragraphs are fed as context to the RoBERTa model, from which it will attempt to extract the answer to your question.  The 'hit' having the highest confidence (prediction probability) from the model is reported as the answer.
    ''')

st.markdown('''
Please provide a question you'd like the model to try to answer.  The model will report back its answer, as well as an excerpt of text from Wikipedia in which it found its answer.  Your result will appear below the question field when the model is finished running.

Alternatively, you can try an example by clicking one of the buttons below:
''')

# Generate containers in order
example_container = st.container()
input_container = st.container()
response_container = st.container()

###########################
### Populate containers ###
###########################

# Populate example button container
with example_container:
    ex_cols = st.columns(len(ex_questions)+1)
    for i in range(len(ex_questions)):
        with ex_cols[i]:
            st.button(
                label = f'example {i+1}',
                key = f'ex_button_{i+1}',
                on_click = fill_in_example,
                args=(i,),
            )
    with ex_cols[-1]:
        st.button(
            label = "Clear all fields",
            key = "clear_button",
            on_click = clear_boxes,
        )

# Populate user input container
with input_container:
    with st.form(key='input_form',clear_on_submit=False):
        # Question input field
        question = st.text_input(
            label='Question',
            value=st.session_state['auto']['question'],
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        # Form submit button
        question_submitted = st.form_submit_button("Submit")
        if question_submitted:
            # update question, context in session state
            st.session_state['auto']['question'] = question
            query = generate_query(nlp,question)
            # query == '' will throw error in document retrieval
            if len(query)==0:
                st.session_state['auto']['response'] = 'Please include some nouns, verbs, and/or adjectives in your question.'
            elif len(question)>0:
                with st.spinner('Retrieving documentation...'):
                    # Retrieve ids from top 3 results
                    retriever = ContextRetriever()
                    retriever.get_pageids(query,topn=3)
                    st.write(retriever.pageids)
                    # Retrieve pages then paragraphs
                    retriever.get_all_pages()
                    retriever.get_all_paragraphs()
                    # Get top 10 paragraphs, ranked by relevance to query
                    best_paragraphs = retriever.rank_paragraphs(retriever.paragraphs, query)

                st.write(best_paragraphs)
                with st.spinner('Generating response...'):
                    # Loop through best_paragraph contexts
                    # looking for answer in each
                    for paragraph in best_paragraphs:
                        input = {
                            'context':paragraph[1],
                            'question':st.session_state['auto']['question'],
                        }
                        # Pass to QA pipeline
                        response = qa_pipeline(**input)
                        # Append answers and scores.  We report a score of 0
                        # for no-answer paragraphs, so they get deprioritized
                        # when the max is taken below
                        if response['answer']!='':
                            paragraph += [response['answer'],response['score']]
                        else:
                            paragraph += ['',0]
                    
                    # Get best paragraph (max confidence score) and collect data
                    best_paragraph = max(best_paragraphs,key = lambda x:x[3])
                    best_answer = best_paragraph[2]
                    best_context_article = best_paragraph[0]
                    best_context = best_paragraph[1]
                    
                    # Update response in session state
                    if best_answer == "":
                        st.session_state['auto']['response'] = "I cannot find the answer to your question."
                    else:
                        st.session_state['auto']['response'] = f"""
                            My answer is: {best_answer}
                    
                            ...and here's where I found it:
                    
                            Article title: {best_context_article}

                            Paragraph containing answer:

                            {best_context}
                        """

# Display response
with response_container:
    st.write(st.session_state['auto']['response'])
            
