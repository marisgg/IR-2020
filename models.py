import itertools
import math
from pyserini.search import SimpleSearcher
import pandas as pd
import collections
from index_trec import Index
import numpy as np
from timer import Timer
from tqdm import tqdm
from pyserini.index import IndexReader

class Models:

    def __init__(self, index, searcher):
        self.index_reader = index
        self.searcher = searcher
        self.N = self.index_reader.stats()['documents']
        self.trec_index = Index(self.index_reader, self.searcher)
        self.t = Timer()

        # global variables for rocchio algorithm
        # so that we only need to compute the document vectors once
        # and only need to adjust the query vector for the new query
        self.c_list = []
        self.relevance_data = None
        self.all_docs= []
        self.top_k_doc_vec = {}
        self.complete_query_vector = []

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




    def create_collection_list(self, top_k_docs):
        """
        Create list of all analyzed terms in the top-k documents.
        """
        for doc in top_k_docs:
            self.top_k_doc_vec[doc] = self.index_reader.get_document_vector(doc)
            for term in self.top_k_doc_vec[doc].keys():
                if not term in self.c_list:
                    self.c_list.append(term)
        print(f"Length of collection terms list: {len(self.c_list)}")
        #for term in itertools.islice(self.index_reader.terms(), 0, None):
        #    self.c_list.append(term.term)
        #    self.c_list = self.c_list[38734:193468] # c_list.index('a.')) #first 38734 elements are numbers, last are mostly symbols


    def create_complete_vector(self, doc_vector):
        """
        Each term is represented as a dimension to be able to compare the documents.
        For this, also the terms that are not present in the document should be added to the document vector.
        """
        complete_doc_vector = doc_vector
        print(f"Length of document vector before completing with collection list: {len(doc_vector)}")
        for term in self.c_list:    
            # TODO: smoothing of terms?
            if not term in doc_vector:
                complete_doc_vector[term] = 0
        complete_dict = collections.OrderedDict(sorted(complete_doc_vector.items()))
        complete_vector_list = complete_dict.values()
        print(f"Length after completing: {len(complete_vector_list)}")
        return complete_vector_list
            

    def create_query_vector(self, q):
        query_vector = {}
        for t in q:
            #if not t in query_vector:
            query_vector[t] = 1 # =/= q.count(t), since BOW uses binary representation of the occurence of a term
        complete_query_vector = self.create_complete_vector(query_vector)
        return complete_query_vector


    def get_relevance_dataframe(self):
        """
        Read file with relevance of part of the collection.
        """
        self.relevance_data = pd.read_csv("qrels-covid_d5_j0.5-5.txt", sep=" ", header=None)
        self.relevance_data.columns = ["topic_id", "round_id", "cord_uid", "relevancy"]

        self.relevance_data = self.relevance_data[self.relevance_data.relevancy >= 0] # File contains 2 rows with -1

    
    def get_relevance_docs(self, query_id, q):
        """
        Relevancy equals to 0 is irrelevant, 1 is relevant, and 2 is highly relevant.
        """
        self.get_relevance_dataframe()

        relevant_docs = self.relevance_data[(self.relevance_data.topic_id == query_id) & (self.relevance_data.relevancy > 1)] # only highly relevant feedback?
        non_relevant_docs = self.relevance_data[(self.relevance_data.topic_id == query_id) & (self.relevance_data.relevancy == 0)] # Only use positive feedback ?

        # Add terms in relevant & non-relevant docs to the collection term list
        # Create complete query vector accordingly.

        self.create_collection_list(relevant_docs.cord_uid)
        self.create_collection_list(non_relevant_docs.cord_uid)
        
        return [relevant_docs.cord_uid, non_relevant_docs.cord_uid]
    

    def rocchio_algorithm(self, qid, q0):
        print("in rocchio algorithm")
        self.t.start()
        doc_ids = self.get_relevance_docs(qid, q0)
        relevant_doc_ids = doc_ids[0]
        non_relevant_doc_ids = doc_ids[1]

        relevant_doc_vectors = []
        count = 0
        for doc_id in tqdm(relevant_doc_ids):
            relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
            count += 1
            break
        print("Got relevant vectors")
        self.t.stop()

        self.t.start()
        non_relevant_doc_vectors = []
        for doc_id in tqdm(non_relevant_doc_ids):
            non_relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
            break
        print("Got non-relevant vectors")
        self.t.stop()

        # Standard values
        beta = 0.75
        alpha = 1.0
        gamma = 0.15

        # calculate centroid of relevant documents
        self.t.start()
        summed_relevant_vectors = np.zeros(len(relevant_doc_vectors[0]))
        for doc in tqdm(relevant_doc_vectors): 
            np.add(summed_relevant_vectors, np.array(list(doc)), summed_relevant_vectors)
        centroid_relevant_docs = 1/len(relevant_doc_ids) * summed_relevant_vectors
        print(f"centroid rel length: {len(centroid_relevant_docs)}")
        self.t.stop()

        self.t.start()
        # TODO: In practice, only use positive feedback (set gamma to 0) --> test this
        # calculate centroid of non-relevant documents
        summed_non_relevant_vectors = np.zeros(len(non_relevant_doc_vectors[0]))
        for doc in tqdm(non_relevant_doc_vectors): 
            summed_non_relevant_vectors = np.add(summed_non_relevant_vectors, np.array(list(doc)), summed_non_relevant_vectors)
        centroid_non_relevant_docs = 1/len(non_relevant_doc_ids) * summed_non_relevant_vectors
        print(f"centroid non-rel length: {len(centroid_non_relevant_docs)}")
        self.t.stop()

        complete_q0 = self.create_query_vector(q0)

        self.t.start()
        # rocchio algorithm    
        q_mod = np.multiply(alpha, np.asarray(list(complete_q0))) + np.multiply(beta, np.asarray(centroid_relevant_docs)) - np.multiply(gamma, np.asarray(centroid_non_relevant_docs))
        self.t.stop() 
        print(q_mod[:300])
        return q_mod

    def rocchio_ranking(self, qid, q0, top_k_docs):
        rocchio_timer = Timer()
        rocchio_timer.start()
        if self.c_list == []:
            # start by adding query terms to the terms collection
            for term in q0:
                self.c_list.append(term)
            # add the terms from the top-k documents
            self.create_collection_list(top_k_docs)
        print("in rocchio ranking")

        self.t.start()
        if self.all_docs == [] :
            self.all_docs = self.trec_index.get_docids() # ~ 30 seconds
        print("got all docs")
        self.t.stop()

        q_mod = self.rocchio_algorithm(qid, q0)
        print("got qmod")
        
        doc_scores = {}

        count = 0

        self.t.start()

        # Rank documents using dot product as similarity function
        for doc in top_k_docs:
            
            similarity_score = np.dot(np.array(list(self.create_complete_vector(self.tf_idf_docid(doc)))), q_mod)
            doc_scores[doc] = similarity_score
            print(f"Doc nr.:{count} - Score: {similarity_score}")
            count += 1
        self.t.stop()
        print(doc_scores)
        rocchio_timer.stop()
        return doc_scores

"""
# code to plot graph with the balance of relevancy per topic.
import plotly.express as px

fig = px.histogram(relevance_data, x="topic_id", color = "relevancy")
fig.show()
"""
