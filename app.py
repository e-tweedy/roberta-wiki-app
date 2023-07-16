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

#################################
### Model retrieval functions ###
#################################

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
for tab in ['basic','semi','auto']:
    if tab not in st.session_state:
        st.session_state[tab] = {}
    for field in ['question','context','query','response']:
        if field not in st.session_state[tab]:
            st.session_state[tab][field] = ''
for field in ['page_options','selected_pages']:
    if field not in st.session_state['semi']:
        st.session_state['semi'][field] = []
        
# Retrieve models
with st.spinner('Loading the model...'):
    qa_pipeline = get_pipeline()
    nlp = get_spacy()

# Retrieve example questions and contexts
examples = get_examples()
# ex_queries, ex_questions, ex_contexts = get_examples()
if 'ex_questions' not in st.session_state['semi']:
    st.session_state['semi']['ex_questions'] = len(examples[1][0])*['']
    
################################
### Initialize App Structure ###
################################

tabs = st.tabs([
    'RoBERTa Q&A model',
    'Basic extractive Q&A',
    'User-guided Wiki Q&A',
    'Automated Wiki Q&A',
])

with tabs[0]:
    intro_container = st.container()
with tabs[1]:
    basic_title_container = st.container()
    basic_example_container = st.container()
    basic_input_container = st.container()
    basic_response_container = st.container()
with tabs[2]:
    semi_title_container = st.container()
    semi_query_container = st.container()
    semi_page_container = st.container()
    semi_input_container = st.container()
    semi_response_container = st.container()
with tabs[3]:
    auto_title_container = st.container()
    auto_example_container = st.container()
    auto_input_container = st.container()
    auto_response_container = st.container()

#####################
### Tab - Welcome ###
#####################

with intro_container:
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
2. A user-guided Wiki Q&A tool which allows the user to search for one or more Wikipedia pages and ask the model to search those pages for the answer to a user-provided question.
3. An automated Wiki Q&A tool which asks the model to perform retrieve its own Wikipedia pages in order to answer the user-provided question.
    ''')

#######################
### Tab - basic Q&A ###
#######################

from lib.utils import basic_clear_boxes, basic_ex_click

with basic_title_container:
    ### Intro text ###
    st.header('Basic extractive Q&A')
    st.markdown('''
The basic functionality of a RoBERTa model for extractive question-answering is to attempt to extract the answer to a user-provided question from a piece of user-provided context text.  The model is also trained to recognize when the context doesn't provide the answer.

Please type or paste a context paragraph and question you'd like to ask about it.  The model will attempt to answer the question based on the context you provided, or report that it cannot find the answer in the context.  Your results will appear below the question field when the model is finished running.

Alternatively, you can try an example by clicking one of the buttons below:
    ''')
    
### Populate example button container ###
with basic_example_container:
    basic_ex_cols = st.columns(len(examples[0])+1)
    for i in range(len(examples[0])):
        with basic_ex_cols[i]:
            st.button(
                label = f'example {i+1}',
                key = f'basic_ex_button_{i+1}',
                on_click = basic_ex_click,
                args = (examples,i,),
            )
    with basic_ex_cols[-1]:
        st.button(
            label = "Clear all fields",
            key = "basic_clear_button",
            on_click = basic_clear_boxes,
        )
### Populate user input container ###
with basic_input_container:
    with st.form(key='basic_input_form',clear_on_submit=False):
        # Context input field
        context = st.text_area(
            label='Context',
            value=st.session_state['basic']['context'],
            key='basic_context_field',
            label_visibility='collapsed',
            placeholder='Enter your context paragraph here.',
            height=300,
        )
        # Question input field
        question = st.text_input(
            label='Question',
            value=st.session_state['basic']['question'],
            key='basic_question_field',
            label_visibility='collapsed',
            placeholder='Enter your question here.',
        )
        # Form submit button
        query_submitted = st.form_submit_button("Submit")
        if query_submitted and question!= '':
            # update question, context in session state
            st.session_state['basic']['question'] = question
            st.session_state['basic']['context'] = context
            with st.spinner('Generating response...'):
                # Generate dictionary from inputs
                query = {
                    'context':st.session_state['basic']['context'],
                    'question':st.session_state['basic']['question'],
                }
                # Pass to QA pipeline
                response = qa_pipeline(**query)
                answer = response['answer']
                confidence = response['score']
                # Reformat empty answer to message
                if answer == '':
                    answer = "I don't have an answer based on the context provided."
                # Update response in session state
                st.session_state['basic']['response'] = f"""
                    Answer: {answer}\n
                    Confidence: {confidence:.2%}
                """
### Populate response container ###
with basic_response_container:
    st.write(st.session_state['basic']['response'])
            
########################
### Tab - guided Q&A ###
########################

from lib.utils import (
    semi_ex_query_click,
    semi_ex_question_click,
    semi_clear_query,
    semi_clear_question,
)
    
### Intro text ###
with semi_title_container:
    st.header('User-guided Wiki Q&A')
    st.markdown('''
This component allows you to perform a Wikipedia search for source material to feed as contexts to the RoBERTa question-answering model.
    ''')
    with st.expander("Click here to find out what's happening behind the scenes..."):
        st.markdown('''
    1. A Wikipedia search is performed using your query, resulting in a list of pages which then populate the drop-down menu.
    2. The pages you select are retrieved and broken up into paragraphs.  Wikipedia queries and page collection use the [wikipedia library](https://pypi.org/project/wikipedia/), a wrapper for the [MediaWiki API](https://www.mediawiki.org/wiki/API).
    3. The paragraphs are ranked in descending order of relevance to your question, using the [Okapi BM25 score](https://en.wikipedia.org/wiki/Okapi_BM25) as implemented in the [rank_bm25 library](https://github.com/dorianbrown/rank_bm25).
    4. Among these ranked paragraphs, approximately the top 25% are fed as context to the RoBERTa model, from which it will attempt to extract the answer to your question.  The 'hit' having the highest confidence (prediction probability) from the model is reported as the answer.
        ''')

### Populate query container ###
with semi_query_container:
    st.markdown('First submit a search query, or choose one of the examples.')
    semi_query_cols = st.columns(len(examples[0])+1)
    # Buttons for query examples
    for i in range(len(examples[0])):
        with semi_query_cols[i]:
            st.button(
                label = f'query {i+1}',
                key = f'semi_query_button_{i+1}',
                on_click = semi_ex_query_click,
                args=(examples,i,),
            )
    # Button for clearning query field
    with semi_query_cols[-1]:
        st.button(
            label = "Clear query",
            key = "semi_clear_query",
            on_click = semi_clear_query,
        )
    # Search query input form
    with st.form(key='semi_query_form',clear_on_submit=False):
        query = st.text_input(
            label='Search query',
            value=st.session_state['semi']['query'],
            key='semi_query_field',
            label_visibility='collapsed',
            placeholder='Enter your Wikipedia search query here.',
        )
        query_submitted = st.form_submit_button("Submit")

        if query_submitted and query != '':
            st.session_state['semi']['query'] = query
            # Retrieve Wikipedia page list from
            # search results and store in session state
            with st.spinner('Retrieving Wiki pages...'):
                retriever = ContextRetriever()
                retriever.get_pageids(query)
                st.session_state['semi']['page_options'] = retriever.pageids
                st.session_state['semi']['selected_pages'] = []
    
### Populate page selection container ###
with semi_page_container:
    st.markdown('Next select any number of Wikipedia pages to provide to RoBERTa:')
    # Page title selection form
    with st.form(key='semi_page_form',clear_on_submit=False):
        selected_pages = st.multiselect(
                label = "Choose Wiki pages for Q&A model:",
                options = st.session_state['semi']['page_options'],
                default = st.session_state['semi']['selected_pages'],
                label_visibility = 'collapsed',
                key = "semi_page_selectbox",
                format_func = lambda x:x[1],
            )
        pages_submitted = st.form_submit_button("Submit")
        if pages_submitted:
            st.session_state['semi']['selected_pages'] = selected_pages

### Populate question input container ###
with semi_input_container:
    st.markdown('Finally submit a question for RoBERTa to answer based on the above pages or choose one of the examples.')
    # Question example buttons
    semi_ques_cols = st.columns(len(examples[0])+1)
    for i in range(len(examples[0])):
        with semi_ques_cols[i]:
            st.button(
                label = f'question {i+1}',
                key = f'semi_ques_button_{i+1}',
                on_click = semi_ex_question_click,
                args=(i,),
            )
    # Question field clear button
    with semi_ques_cols[-1]:
        st.button(
            label = "Clear question",
            key = "semi_clear_question",
            on_click = semi_clear_question,
        )
    # Question submission form
    with st.form(key = "semi_question_form",clear_on_submit=False):
        question = st.text_input(
            label='Question',
            value=st.session_state['semi']['question'],
            key='semi_question_field',
            label_visibility='collapsed',
            placeholder='Enter your question here.',
        )
        question_submitted = st.form_submit_button("Submit")
        if question_submitted and len(question)>0 and len(st.session_state['semi']['selected_pages'])>0:
            st.session_state['semi']['response'] = ''
            st.session_state['semi']['question'] = question
            # Retrieve pages corresponding to user selections,
            # extract paragraphs, and retrieve top 10 paragraphs,
            # ranked by relevance to user question
            with st.spinner("Retrieving documentation..."):
                retriever = ContextRetriever()
                pages = retriever.ids_to_pages(selected_pages)
                paragraphs = retriever.pages_to_paragraphs(pages)
                best_paragraphs = retriever.rank_paragraphs(
                    paragraphs, question,
                )     
            with st.spinner("Generating response..."):
                # For each paragraph, format input to QA pipeline...
                for paragraph in best_paragraphs:
                    input = {
                        'context':paragraph[1],
                        'question':st.session_state['semi']['question'],
                    }
                    # ...and pass to QA pipeline
                    response = qa_pipeline(**input)
                    # Append answers and scores.  Report score of
                    # zero for paragraphs without answer, so they are
                    # deprioritized when the max is taken below
                    if response['answer']!='':
                        paragraph += [response['answer'],response['score']]               
                    else:
                        paragraph += ['',0]
                # Get paragraph with max confidence score and collect data
                best_paragraph = max(best_paragraphs,key = lambda x:x[3])
                best_answer = best_paragraph[2]
                best_context_page = best_paragraph[0]
                best_context = best_paragraph[1]
                
                # Update response in session state
                if best_answer == "":
                    st.session_state['semi']['response'] = "I cannot find the answer to your question."
                else:
                    st.session_state['semi']['response'] = f"""
                    My answer is: {best_answer}
                    
                    ...and here's where I found it:
                    
                    Page title: {best_context_page}

                    Paragraph containing answer:

                    {best_context}
                    """
### Populate response container ###
with semi_response_container:
    st.write(st.session_state['semi']['response'])

###########################
### Tab - automated Q&A ###
###########################

from lib.utils import auto_ex_click, auto_clear_boxes
    
### Intro text ###
with auto_title_container:
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

### Populate example container ###
with auto_example_container:
    auto_ex_cols = st.columns(len(examples[0])+1)
    # Buttons for selecting example questions
    for i in range(len(examples[0])):
        with auto_ex_cols[i]:
            st.button(
                label = f'example {i+1}',
                key = f'auto_ex_button_{i+1}',
                on_click = auto_ex_click,
                args=(examples,i,),
            )
    # Button for clearing question field and response
    with auto_ex_cols[-1]:
        st.button(
            label = "Clear all fields",
            key = "auto_clear_button",
            on_click = auto_clear_boxes,
        )

### Populate user input container ###
with auto_input_container:
    with st.form(key='auto_input_form',clear_on_submit=False):
        # Question input field
        question = st.text_input(
            label='Question',
            value=st.session_state['auto']['question'],
            key='auto_question_field',
            label_visibility='collapsed',                
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
                    # Retrieve pages then paragraphs
                    retriever.get_all_pages()
                    retriever.get_all_paragraphs()
                    # Get top 10 paragraphs, ranked by relevance to query
                    best_paragraphs = retriever.rank_paragraphs(retriever.paragraphs, query)
                with st.spinner('Generating response...'):
                        # For each paragraph, format input to QA pipeline...
                    for paragraph in best_paragraphs:
                        input = {
                            'context':paragraph[1],
                            'question':st.session_state['auto']['question'],
                        }
                        # ...and pass to QA pipeline
                        response = qa_pipeline(**input)
                        # Append answers and scores.  Report score of
                        # zero for paragraphs without answer, so they are
                        # deprioritized when the max is taken below
                        if response['answer']!='':
                            paragraph += [response['answer'],response['score']]
                        else:
                            paragraph += ['',0]
                    
                    # Get paragraph with max confidence score and collect data
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
### Populate response container ###
with auto_response_container:
    st.write(st.session_state['auto']['response'])