import itertools
import math

class Models:

    def __init__(self, index, searcher):
        self.index_reader = index
        self.searcher = searcher
        self.N = self.index_reader.stats()['documents']
        self.doc = None

    def set_doc(self, doc):
        self.doc = doc

    def get_n_of_words_in_docid(self, docid):
        """ Hacky: Sum all term frequencies in document vector (thus no stopwords) """
        return sum(self.index_reader.get_document_vector(docid).values())

    """
    Compute TF-IDF, which consists of the following two components:
    1. Term frequency: measures the frequency of a word in a document, normalize.
        tf(t,d) = count of t in d / number of words in d
    2. Inverse document frequency: measures the informativeness of term t.
        idf(t) = log(N / (df + 1)               (df = occurences of t in documents)

    The resulting formula: tf-idf(t,d) = tf(t,d)*log(N/(df+1))

    INPUT:      Dictionary, with for each file a sub-dictionary containing the title, abstract, and introduction.
    OUTPUT:     
    """

    def tf_idf_term(self, term, docid) -> float:
        tfs = self.index_reader.get_document_vector(docid)
        if term in tfs:
            tf = tfs[term]/self.get_n_of_words_in_docid(docid)
            df = self.index_reader.get_term_counts(term, analyzer=None)[0]
            return tf * math.log(self.N / (df + 1))
        else:
            return 0.0

    def tf_idf_docid(self, docid) -> {}:
        tfs = self.index_reader.get_document_vector(docid)
        tf_idf = {}
        for term, count in tfs.items():
            df = self.index_reader.get_term_counts(term, analyzer=None)[0]
            tf_idf[term] = count/self.get_n_of_words_in_docid(docid) * math.log(self.N / (df + 1)) # added total number of words in doc
        return tf_idf

    def bm25_term(self, docid, term) -> float:
        return self.index_reader.compute_bm25_term_weight(docid, term, analyzer=None)

    def bm25(self, query, docid) -> float:
        return self.index_reader.compute_bm25_term_weight(docid, query)

    def bm25_mapper(self, term):
        return self.index_reader.compute_bm25_term_weight(self.doc, term)

    def bm25_query_score(self, docid, query):
        return sum([self.index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in query])

    def bm25_docid(self, docid) -> {}:
        """ get all terms in documents """
        tfs = self.index_reader.get_document_vector(docid)
        bm25_vector = {term: self.index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in tfs.keys()}
        return bm25_vector

    def bm25_docid_query(self, docid, query) -> float:
        vec = self.bm25_docid(docid)
        score = 0.0
        for term in query:
            try:
                score += vec[term]
            except:
                pass
        return score