import torch
import streamlit as st
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)
from lib.utils import get_examples

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

def clear_boxes():
    """
    Clears the question, context, response
    """
    for field in ['question','context','response']:
        st.session_state['basic'][field] = ''

def example_click(i):
    """
    Fills in the chosen example
    """
    st.session_state['basic']['question'] = ex_questions[i]
    st.session_state['basic']['context'] = ex_contexts[i]

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
if 'basic' not in st.session_state:
    st.session_state['basic'] = {}
for field in ['question','context','response']:
    if field not in st.session_state['basic']:
        st.session_state['basic'][field] = ''

# Retrieve stored model
with st.spinner('Loading the model...'):
    qa_pipeline = get_pipeline()

# Retrieve example questions and contexts
_, ex_questions, ex_contexts = get_examples()
ex_questions = [q[0] for q in ex_questions]

###################
### App content ###
###################

# Intro text
st.header('Basic extractive Q&A')
st.markdown('''
The basic functionality of a RoBERTa model for extractive question-answering is to attempt to extract the answer to a user-provided question from a piece of user-provided context text.  The model is also trained to recognize when the context doesn't provide the answer.

Please type or paste a context paragraph and question you'd like to ask about it.  The model will attempt to answer the question based on the context you provided, or report that it cannot find the answer in the context.  Your results will appear below the question field when the model is finished running.

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
                label = f'Try example {i+1}',
                key = f'ex_button_{i+1}',
                on_click = example_click,
                args = (i,),
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
        # Context input field
        context = st.text_area(
            label='Context',
            value=st.session_state['basic']['context'],
            key='context_field',
            label_visibility='hidden',
            placeholder='Enter your context paragraph here.',
            height=300,
        )
        # Question input field
        question = st.text_input(
            label='Question',
            value=st.session_state['basic']['question'],
            key='question_field',
            label_visibility='hidden',
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
# Display response
with response_container:
    st.write(st.session_state['basic']['response'])
            
