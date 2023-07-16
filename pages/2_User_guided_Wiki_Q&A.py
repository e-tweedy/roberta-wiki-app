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

def clear_query():
    """
    Clears the search query field
    and page options list
    """
    st.session_state['semi']['query'] = ''
    st.session_state['semi']['page_options'] = []

def clear_question():
    """
    Clears the question and response field
    and selected pages list
    """
    for field in ['question','response']:
        st.session_state['semi'][field] = ''
    st.session_state['semi']['selected_pages'] = []

def select_pages():
    """
    Sets the chosen page titles in the session state
    """
    st.session_state['semi']['selected_pages'] = selected_pages

def query_example_click(i):
    """
    Fills in the query example and
    populates question examples when query
    example button is clicked
    """
    st.session_state['semi']['query'] = ex_queries[i]
    st.session_state['semi']['ex_questions'] = ex_questions[i]

def question_example_click(i):
    """
    Fills in the question example
    """
    st.session_state['semi']['question'] = st.session_state['semi']['ex_questions'][i]


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
if 'semi' not in st.session_state:
    st.session_state['semi'] = {}
for field in ['question','query','response']:
    if field not in st.session_state['semi']:
        st.session_state['semi'][field] = ''
if 'disable_selectbox' not in st.session_state['semi']:
    st.session_state['semi']['disable_selectbox'] = True
for field in ['page_options','selected_pages']:
    if field not in st.session_state['semi']:
        st.session_state['semi'][field] = []

# Retrieve trained RoBERTa pipeline for Q&A
with st.spinner('Loading the model...'):
    qa_pipeline = get_pipeline()

# Retrieve example queries and questions
ex_queries, ex_questions, _ = get_examples()
if 'ex_questions' not in st.session_state['semi']:
    st.session_state['semi']['ex_questions'] = len(ex_questions[0])*['']

###################
### App content ###
###################

# # Intro text
st.header('User-guided Wiki Q&A')
st.markdown('''
This component allows you to perform a Wikipedia search for source material to feed as contexts to the RoBERTa question-answering model.
''')
with st.expander("Click here to find out what's happening behind the scenes..."):
    st.markdown('''
    1. A Wikipedia search is performed using your query, resulting in a list of articles which then populate the drop-down menu.
    2. The articles you select are retrieved and broken up into paragraphs.  Wikipedia queries and article collection use the [wikipedia library](https://pypi.org/project/wikipedia/), a wrapper for the [MediaWiki API](https://www.mediawiki.org/wiki/API).
    3. The paragraphs are ranked in descending order of relevance to your question, using the [Okapi BM25 score](https://en.wikipedia.org/wiki/Okapi_BM25) as implemented in the [rank_bm25 library](https://github.com/dorianbrown/rank_bm25).
    4. Among these ranked paragraphs, approximately the top 25% are fed as context to the RoBERTa model, from which it will attempt to extract the answer to your question.  The 'hit' having the highest confidence (prediction probability) from the model is reported as the answer.
    ''')

# Generate containers in order
query_container = st.container()

article_container = st.container()

input_container = st.container()

response_container = st.container()

###########################
### Populate containers ###
###########################

with query_container:
    st.markdown('First submit a search query, or choose one of the examples.')
    query_cols = st.columns(len(ex_queries)+1)
    for i in range(len(ex_questions)):
        with query_cols[i]:
            st.button(
                label = f'query {i+1}',
                key = f'query_button_{i+1}',
                on_click = query_example_click,
                args=(i,),
            )
    with query_cols[-1]:
        st.button(
            label = "Clear query",
            key = "clear_query",
            on_click = clear_query,
        )
    with st.form(key='query_form',clear_on_submit=False):
        # Search query input field
        query = st.text_input(
            label='Search query',
            value=st.session_state['semi']['query'],
            key='query_field',
            label_visibility='hidden',
            placeholder='Enter your Wikipedia search query here.',
        )
        # Form submit button
        query_submitted = st.form_submit_button("Submit")

        if query_submitted and query != '':
            # update question, context in session state
            st.session_state['semi']['query'] = query
            with st.spinner('Retrieving Wiki articles...'):
                retriever = ContextRetriever()
                retriever.get_pageids(query)
                st.session_state['semi']['page_options'] = retriever.pageids
                st.session_state['semi']['selected_pages'] = []

with article_container:
    # Page title selection box
    st.markdown('Next select any number of Wikipedia articles to provide to RoBERTa:')
    selected_pages = st.multiselect(
            label = "Choose Wiki articles for Q&A model:",
            options = st.session_state['semi']['page_options'],
            default = st.session_state['semi']['selected_pages'],
            key = "page_selectbox",
            format_func = lambda x:x[1],
            on_change = select_pages,
        )

with input_container:
    st.markdown('Finally submit a question for RoBERTa to answer based on the above articles or choose one of the examples.')
    # Question example buttons
    ques_cols = st.columns(len(ex_questions[0])+1)
    for i in range(len(ex_questions)):
        with ques_cols[i]:
            st.button(
                label = f'question {i+1}',
                key = f'ques_button_{i+1}',
                on_click = question_example_click,
                args=(i,),
            )
    with ques_cols[-1]:
        st.button(
            label = "Clear question",
            key = "clear_question",
            on_click = clear_question,
        )
    with st.form(key = "question_form",clear_on_submit=False):
        # Question submission field
        question = st.text_input(
            label='Question',
            value=st.session_state['semi']['question'],
            key='question_field',
            label_visibility='hidden',
            placeholder='Enter your question here.',
        )
        question_submitted = st.form_submit_button("Submit")
        if question_submitted and len(question)>0 and len(st.session_state['semi']['selected_pages'])>0:
            st.session_state['semi']['response'] = ''
            st.session_state['semi']['question'] = question
            with st.spinner("Retrieving documentation..."):
                retriever = ContextRetriever()
                pages = retriever.ids_to_pages(selected_pages)
                paragraphs = retriever.pages_to_paragraphs(pages)
                best_paragraphs = retriever.rank_paragraphs(
                    paragraphs, question,
                    topn=None,
                )
            with st.spinner("Generating response..."):
                for paragraph in best_paragraphs:
                    input = {
                        'context':paragraph[1],
                        'question':st.session_state['semi']['question'],
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
                    st.session_state['semi']['response'] = "I cannot find the answer to your question."
                else:
                    st.session_state['semi']['response'] = f"""
                    My answer is: {best_answer}
                    
                    ...and here's where I found it:
                    
                    Article title: {best_context_article}

                    Paragraph containing answer:

                    {best_context}
                    """
with response_container:
    st.write(st.session_state['semi']['response'])