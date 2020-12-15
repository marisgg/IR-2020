import itertools
import math
from pyserini.search import SimpleSearcher
import pandas as pd
import collections
from index_trec import Index
import numpy as np

class Models:

    def __init__(self, index, searcher):
        self.index_reader = index
        self.searcher = searcher
        self.N = self.index_reader.stats()['documents']

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

    def bm25_term(self, term, docid) -> float:
        return self.index_reader.compute_bm25_term_weight(docid, term, analyzer=None)

    def bm25_docid(self, docid) -> {}:
        """ get all terms in documents """
        tfs = self.index_reader.get_document_vector(docid)
        bm25_vector = {term: self.index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in tfs.keys()}
        return bm25_vector




    def create_collection_list(self):
        """
        Create list of all analyzed terms in the collection.
        """
        c_list = []
        for term in itertools.islice(self.index_reader.terms(), 0, None):
            c_list.append(term.term)
        return c_list


    def create_complete_vector(self, doc_vector):
        """
        Each term is represented as a dimension to be able to compare the documents.
        For this, also the terms that are not present in the document should be added to the document vector.
        """
        c_terms = self.create_collection_list()
        complete_doc_vector = doc_vector
        for term in c_terms:
            # TODO: smoothing of terms?
            if not term in doc_vector:
                complete_doc_vector[term] = 0
        complete_dict =  collections.OrderedDict(sorted(complete_doc_vector.items()))
        complete_vector_list = complete_dict.values()
        return complete_vector_list
            

    def create_query_vector(self, q):
        query_vector = {}
        for t in q:
            #if not t in query_vector:
            query_vector[t] = 1 # =/= q.count(t), since BOW uses binary representation of the occurence of a term
        complete_query_vector = self.create_complete_vector(query_vector)
        return complete_query_vector

    
    def get_relevance_docs(self, query_id):
        """
        Read file with relevance of part of the collection.
        Relevancy equals to 0 is irrelevant, 1 is relevant, and 2 is highly relevant.
        """
        relevance_data = pd.read_csv("qrels-covid_d5_j0.5-5.txt", sep=" ", header=None)
        relevance_data.columns = ["topic_id", "round_id", "cord_uid", "relevancy"]

        relevance_data = relevance_data[relevance_data.relevancy >= 0] # File contains 2 rows with -1

        relevant_docs = relevance_data[(relevance_data.topic_id == query_id) & (relevance_data.relevancy > 0)]
        non_relevant_docs = relevance_data[relevance_data.topic_id == 0] # Only use positive feedback

        return [relevant_docs.cord_uid, non_relevant_docs.cord_uid]
    

    def rocchio_algorithm(self, qid, q0):
        doc_ids = self.get_relevance_docs(qid)
        relevant_doc_ids = doc_ids[0]
        non_relevant_doc_ids = doc_ids[1]

        relevant_doc_vectors = []
        for doc_id in relevant_doc_ids:
            relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
        
        non_relevant_doc_vectors = []
        for doc_id in non_relevant_doc_ids:
            non_relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
        
        # Standard values
        beta = 0.75
        alpha = 1.0
        gamma = 0.15

        # calculate centroid of relevant documents
        summed_relevant_vectors = []
        for doc in relevant_doc_vectors: 
            summed_relevant_vectors = np.add(summed_relevant_vectors, doc)
        centroid_relevant_docs = 1/len(relevant_doc_ids) * summed_relevant_vectors
        
        # TODO: In practice, only use positive feedback (set gamma to 0) --> test this
        # calculate centroid of non-relevant documents
        summed_non_relevant_vectors = []
        for doc in non_relevant_doc_vectors: 
            summed_non_relevant_vectors = np.add(summed_relevant_vectors, doc)
        centroid_non_relevant_docs = 1/len(non_relevant_doc_ids) * summed_non_relevant_vectors
        
        # rocchio algorithm    
        q_mod = alpha * q0 + beta * centroid_relevant_docs - gamma * centroid_non_relevant_docs 
        return q_mod

    def rocchio_ranking(self, qid, q0):
        trec_index = Index(self.index_reader, self.searcher)
        all_docs = trec_index.get_docids()

        q_mod = self.rocchio_algorithm(qid, q0)
        
        doc_scores = {}

        count=0
        # Rank documents using dot product as similarity function
        for doc in all_docs:
            similarity_score = np.dot(self.create_complete_vector(self.tf_idf_docid(doc)), q_mod)
            doc_scores[doc] = similarity_score
            print(f"Doc nr.:{count} - Score: {similarity_score}")
            count += 1
        
        return doc_scores

"""
# code to plot graph with the balance of relevancy per topic.
import plotly.express as px

fig = px.histogram(relevance_data, x="topic_id", color = "relevancy")
fig.show()
"""
