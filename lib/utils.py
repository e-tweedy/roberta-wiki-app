import requests, wikipedia, re
from rank_bm25 import BM25Okapi
import streamlit as st
import pandas as pd
import spacy

####################################
## Streamlit app helper functions ##
####################################

def get_examples():
    """
    Function for loading example questions
    and contexts from examples.csv
    Parameters: None
    -----------
    Returns:
    --------
    ex_queries, ex_questions, ex_contexts : list(str), list(list(str)), list(str)
        Example search query, question, and context strings
        (each entry of ex_questions is a list of three question strings)
    """
    examples = pd.read_csv('examples.csv')
    ex_questions = [q.split(':') for q in list(examples['question'])]
    ex_contexts = list(examples['context'])
    ex_queries = list(examples['query'])
    return ex_queries, ex_questions, ex_contexts

def basic_clear_boxes():
    """
    Clears the question, context, response
    """
    for field in ['question','context','response']:
        st.session_state['basic'][field] = ''

def basic_ex_click(examples, i):
    """
    Fills in the chosen example
    """
    st.session_state['basic']['question'] = examples[1][i][0]
    st.session_state['basic']['context'] = examples[2][i]

def semi_clear_query():
    """
    Clears the search query field
    and page options list
    """
    st.session_state['semi']['query'] = ''
    for field in ['selected_pages','page_options']:
        st.session_state['semi'][field] = []

def semi_clear_question():
    """
    Clears the question and response field
    and selected pages list
    """
    for field in ['question','response']:
        st.session_state['semi'][field] = ''

def semi_ex_query_click(examples,i):
    """
    Fills in the query example and
    populates question examples when query
    example button is clicked
    """
    st.session_state['semi']['query'] = examples[0][i]
    st.session_state['semi']['ex_questions'] = examples[1][i]

def semi_ex_question_click(i):
    """
    Fills in the question example
    """
    st.session_state['semi']['question'] = st.session_state['semi']['ex_questions'][i]

def auto_clear_boxes():
    """
    Clears the response and question fields
    """
    for field in ['question','response']:
        st.session_state['auto'][field]=''

def auto_ex_click(examples,i):
    """
    Fills in the chosen example question
    """
    st.session_state['auto']['question'] = examples[1][i][0]

###########################
## Query helper function ##
###########################

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
        
##############################
## Document retriever class ##
##############################

class ContextRetriever:
    """
    Retrieves documents from Wikipedia based on a query,
    and prepared context paragraphs for a RoBERTa model
    """
    def __init__(self,url='https://en.wikipedia.org/w/api.php'):
        self.url = url
        self.pageids = None
        self.pages = None
        self.paragraphs = None
        
    def get_pageids(self,query,topn = None):
        """
        Retrieve page ids corresponding to a search query
        Parameters:
        -----------
        query : str
            A query to use for Wikipedia page search
        topn : int or None
            If topn is provided, will only return pageids
            for topn search results
        Returns: None, but stores:
        --------
        self.pageids : list(tuple(int,str))
            A list of Wikipedia (pageid,title) tuples resulting
            from the search
        """
        params = {
            'action':'query',
            'list':'search',
            'srsearch':query,
            'format':'json',
        }
        results = requests.get(self.url, params=params).json()
        pageids = [(page['pageid'],page['title']) for page in results['query']['search']]
        pageids = pageids[:topn]
        self.pageids = pageids

    def ids_to_pages(self,ids):
        """
        Use MediaWiki API to retrieve page content corresponding to
        a list of pageids
        Parameters:
        -----------
        ids : list(tuple(int,str))
            A list of Wikipedia (pageid,title) tuples
        Returns: None, but stores
        --------
        pages : list(tuple(str,str))
            The k-th enry is a tuple consisting of the title and page content
            of the page corresponding to the k-th entry of ids
        """
        pages = []
        for pageid in ids:
            try:
                page = wikipedia.page(pageid=pageid[0],auto_suggest=False)
                pages.append((page.title, page.content))
            except wikipedia.DisambiguationError:
                continue
        return pages
    
    def get_all_pages(self):
        """
        Use MediaWiki API to retrieve page content corresponding to
        the list of pageids in self.pageids
        Parameters: None
        -----------
        Returns: None, but stores
        --------
        self.pages : list(tuple(str,str))
            The k-th enry is a tuple consisting of the title and page content
            of the page corresponding to the k-th entry of self.pageids
        """
        assert self.pageids is not None, "No pageids exist. Get pageids first using self.get_pageids"
        self.pages = self.ids_to_pages(self.pageids)

    def pages_to_paragraphs(self,pages):
        """
        Process a list of pages into a list of paragraphs from those pages
        Parameters:
        -----------
        pages : list(str)
            A list of Wikipedia page content dumps, as strings
        Returns:
        --------
        paragraphs : dict
            keys are titles of pages from pages (as strings)
            paragraphs[page] is a list of paragraphs (as strings)
            extracted from page
        """
        # Content from WikiMedia has these headings. We only grab content appearing
        # before the first instance of any of these
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        pattern = re.compile(pattern)
        paragraphs = {}
        for page in pages:
            # Truncate page to the first index of the start of a matching heading,
            # or the end of the page if no matches exist
            title, content = page
            idx = min([match.start() for match in pattern.finditer(content)]+[len(content)])
            content = content[:idx]
            # Split into paragraphs, omitting lines with headings (start with '='),
            # empty lines, or lines like '\t\t' or '\t\t\t' which sometimes appear
            paragraphs[title] = [
                p for p in content.split('\n') if p \
                and not p.startswith('=') \
                and not p.startswith('\t\t') \
                and not p.startswith('  ')
            ]
        return paragraphs
    
    def get_all_paragraphs(self):
        """
        Process self.pages into list of paragraphs from pages
        Parameters: None
        -----------
        Returns: None, but stores
        --------
        self.paragraphs : dict
            keys are titles of pages from self.pages (as strings)
            self.paragraphs[page] is a list of paragraphs (as strings)
            extracted from page
        """
        assert self.pages is not None, "No page content exists. Get pages first using self.get_pages"
        # Content from WikiMedia has these headings. We only grab content appearing
        # before the first instance of any of these
        self.paragraphs = self.pages_to_paragraphs(self.pages)

    def rank_paragraphs(self,paragraphs,query,topn=10):
        """
        Ranks the elements of paragraphs in descending order
        by relevance to query using BM25 Okapi, and returns top
        topn results
        Parameters:
        -----------
        paragraphs : dict
            keys are titles of pages (as strings)
            paragraphs[page] is a list of paragraphs (as strings)
            extracted from page
        query : str
            The query to use in ranking paragraphs by relevance
        topn : int or None
            The number of most relevant paragraphs to return
            If None, will return roughly the top 1/4 of the
            paragraphs
        Returns:
        --------
        best_paragraphs : list(list(str,str))
            The k-th entry is a list [title,paragraph] for the k-th
            most relevant paragraph, where title is the title of the
            Wikipedia article from which that paragraph was sourced
        """
        corpus, titles, page_nums = [],[],[]
        # Compile paragraphs into corpus
        for i,page in enumerate(paragraphs):
            titles.append(page)
            paras = paragraphs[page]
            corpus += paras
            page_nums += len(paras)*[i]

        # Tokenize corpus and query and initialize bm25 object
        tokenized_corpus = [p.split(" ") for p in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")

        # Compute scores and compile tuples (paragraph number, score, page number)
        # before sorting tuples by score
        bm_scores = bm25.get_scores(tokenized_query)
        paragraph_data = [[i,score,page_nums[i]] for i,score in enumerate(bm_scores)]
        paragraph_data.sort(reverse=True,key=lambda p:p[1])
        
        # Grab topn best [title,paragraph] pairs sorted by bm25 score
        topn = len(paragraph_data)//4+1 if topn is None else min(topn,len(paragraph_data))

        best_paragraphs = [[titles[p[2]],corpus[p[0]]] for p in paragraph_data[:topn]]
        return best_paragraphs

def generate_answer(pipeline,paragraphs, question):
    """
    Generate an answer using a question-answer pipeline
    Parameters:
    -----------
    pipeline : transformers.QuestionAnsweringPipeline
        The question answering pipeline object
    paragraphs : list(list(str,str))
        The k-th entry is a list [title,paragraph] consisting
        of a context paragraph and the title of the page from which the
        paragraph was sourced
    question : str
        A question that is to be answered based on context given
        in the entries of paragraphs

    Returns:
    --------
    response : str
        A response indicating the answer that was discovered,
        or indicating that no answer could be found.
    """
    # For each paragraph, format input to QA pipeline...
    for paragraph in paragraphs:
        input = {
            'context':paragraph[1],
            'question':question,
        }
        # ...and pass to QA pipeline
        output = pipeline(**input)
        # Append answers and scores.  Report score of
        # zero for paragraphs without answer, so they are
        # deprioritized when the max is taken below
        if output['answer']!='':
            paragraph += [output['answer'],output['score']]               
        else:
            paragraph += ['',0]
    # Get paragraph with max confidence score and collect data
    best_paragraph = max(paragraphs,key = lambda x:x[3])
    best_answer = best_paragraph[2]
    best_context_page = best_paragraph[0]
    best_context = best_paragraph[1]
                
    # Update response in session state
    if best_answer == "":
        response = "I cannot find the answer to your question."
    else:
        response = f"""
            My answer is: {best_answer}
                    
            ...and here's where I found it:
                    
            Page title: {best_context_page}

            Paragraph containing answer:

            {best_context}
            """
    return response