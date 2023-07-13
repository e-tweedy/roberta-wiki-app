import requests, wikipedia, re, spacy
from rank_bm25 import BM25Okapi
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)

class QueryProcessor:
    """
    Processes text into queries using a spaCy model
    """
    def __init__(self):
        self.keep = {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}
        self.nlp = spacy.load(
            'en_core_web_sm',
            disable = ['ner','parser','textcat']
        )

    def generate_query(self,text):
        """
        Process text into a search query,
        only retaining nouns, proper nouns numerals, verbs, and adjectives
        Parameters:
        -----------
        text : str
            The input text to be processed
        Returns:
        --------
        query : str
            The condensed search query
        """
        tokens = self.nlp(text)
        query = ' '.join(token.text for token in tokens \
                         if token.pos_ in self.keep)
        return query

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
        
    def get_pageids(self,query):
        """
        Retrieve page ids corresponding to a search query
        Parameters:
        -----------
        query : str
            A query to use for Wikipedia page search
        Returns: None, but stores:
        --------
        self.pageids : list(int)
            A list of Wikipedia page ids corresponding to search results
        """
        params = {
            'action':'query',
            'list':'search',
            'srsearch':query,
            'format':'json',
        }
        results = requests.get(self.url, params=params).json()
        pageids = [page['pageid'] for page in results['query']['search']]
        self.pageids = pageids
    
    def get_pages(self):
        """
        Use MediaWiki API to retrieve page content corresponding to
        entries of self.pageids
        Parameters: None
        -----------
        Returns: None, but stores
        --------
        self.pages : list(str)
            Entries are content of pages corresponding to entries of self.pageid
        """
        assert self.pageids is not None, "No pageids exist. Get pageids first using self.get_pageids"
        self.pages = []
        for pageid in self.pageids:
            try:
                self.pages.append(wikipedia.page(pageid=pageid,auto_suggest=False).content)
            except wikipedia.DisambiguationError as e:
                continue

    def get_paragraphs(self):
        """
        Process self.pages into list of paragraphs from pages
        Parameters: None
        -----------
        Returns: None, but stores
        --------
        self.paragraphs : list(str)
            List of paragraphs from all pages in self.pages, in order of self.pages
        """
        assert self.pages is not None, "No page content exists. Get pages first using self.get_pages"
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
        paragraphs = []
        for page in self.pages:
            # Truncate page to the first index of the start of a matching heading,
            # or the end of the page if no matches exist
            idx = min([match.start() for match in pattern.finditer(page)]+[len(page)])
            page = page[:idx]
            # Split into paragraphs, omitting lines with headings (start with '='),
            # empty lines, or lines like '\t\t' or '\t\t\t' which sometimes appear
            paragraphs += [
                p for p in page.split('\n') if p \
                and not p.startswith('=') \
                and not p.startswith('\t\t')
            ]
        self.paragraphs = paragraphs

    def rank_paragraphs(self,query,topn=10):
        """
        Ranks the elements of self.paragraphs in descending order
        by relevance to query using BM25F, and returns top topn results
        Parameters:
        -----------
        query : str
            The query to use in ranking paragraphs by relevance
        topn : int
            The number of most relevant paragraphs to return
        Returns: None, but stores
        --------
        self.best_paragraphs : list(str)
            The topn most relevant paragraphs to the query
        """
        tokenized_paragraphs = [p.split(" ") for p in self.paragraphs]
        bm25 = BM25Okapi(tokenized_paragraphs)
        tokenized_query = query.split(" ")
        self.best_paragraphs = bm25.get_top_n(tokenized_query,self.paragraphs,n=topn)

