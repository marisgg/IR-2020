import itertools
import math
import pandas as pd
import collections
from index_trec import Index
import numpy as np
from timer import Timer
from tqdm import tqdm

class Models:

    def __init__(self, index, qrelfile):
        self.index_reader = index
        self.N = self.index_reader.stats()['documents']
        self.t = Timer()

        self.qrelfile = qrelfile
        self.df_vector = {}

        # global variables for rocchio algorithm
        # so that we only need to compute the document vectors once
        # and only need to adjust the query vector for the new query
        self.c_list = []
        self.relevance_data = None
        self.all_docs= []
        self.top_k_doc_vec = {}
        self.complete_query_vector = []

    def compute_df_vector(self, term):
        self.df_vector[term] = self.index_reader.get_term_counts(term, analyzer=None)[0]

    def reset_df_vector(self):
        self.df_vector = {}

    def get_n_of_words_in_docid(self, docid):
        """ Hacky: Sum all term frequencies in document vector (thus no stopwords) """
        return sum(self.index_reader.get_document_vector(docid).values())

    def docid_length(self, docid):
        return len([item for sublist in list(self.index_reader.get_term_positions(docid).values()) for item in sublist])

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

    def tf_idf_term(self, docid, term, use_vector=True, wordcount=None, tfs=None) -> float:
        try:
            # Might throw keyerror, then return 0.0 (doesn't exist)
            if tfs is None:
                tfs = self.index_reader.get_document_vector(docid)
            if wordcount is None:
                wordcount = self.get_n_of_words_in_docid(docid)
            tf = tfs[term] / wordcount
            if use_vector:
                df = self.df_vector[term]
            else:
                df = self.index_reader.get_term_counts(term, analyzer=None)[0]
            return tf * math.log(self.N / (df + 1))
        except KeyError:
            return 0.0

    def tf_idf_docid(self, docid, wordcount=None) -> {}:
        tfs = self.index_reader.get_document_vector(docid)
        tf_idf = {}
        if wordcount is None:
            wordcount = self.get_n_of_words_in_docid(docid)
        for term, count in tfs.items():
            df = self.index_reader.get_term_counts(term, analyzer=None)[0]
            tf_idf[term] = (count / wordcount) * math.log(self.N / (df + 1)) # added total number of words in doc
        return tf_idf

    def tf_idf_query(self, docid, query) -> float:
        tfs = self.index_reader.get_document_vector(docid)
        wordcount = self.get_n_of_words_in_docid(docid)
        return sum([self.tf_idf_term(docid, term, wordcount=wordcount, tfs=tfs) for term in query])

    def bm25_term(self, docid, term, k1=0.9, b=0.4) -> float:
        return self.index_reader.compute_bm25_term_weight(docid, term, k1=k1, b=b, analyzer=None)

    def bm25_query_score(self, docid, query, k1=0.9, b=0.4) -> float:
        return sum([self.bm25_term(docid, term, k1=k1, b=b) for term in query])

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
        #print(f"Length of document vector before completing with collection list: {len(doc_vector)}")
        for term in self.c_list:    
            # TODO: smoothing of terms?
            if not term in doc_vector:
                complete_doc_vector[term] = 0
        complete_dict = collections.OrderedDict(sorted(complete_doc_vector.items()))
        complete_vector_list = complete_dict.values()
        #print(f"Length after completing: {len(complete_vector_list)}")
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
        self.relevance_data = pd.read_csv(self.qrelfile, sep=" ", header=None)
        self.relevance_data.columns = ["topic_id", "round_id", "cord_uid", "relevancy"]

        self.relevance_data = self.relevance_data[self.relevance_data.relevancy >= 0] # File contains 2 rows with -1

    
    def get_relevance_docs(self, query_id, q, m, ordered_doc_scores):
        """
        Relevancy equals to 0 is irrelevant, 1 is relevant, and 2 is highly relevant.
        Ide dec-h algoritm: Take only the marked non-relevant document that received the highest score.
        """
        self.get_relevance_dataframe()

        relevant_docs = self.relevance_data[(self.relevance_data.topic_id == query_id) & (self.relevance_data.relevancy > 1)] # only highly relevant feedback?
        if m == 'ide':
            # Loop over ordered dict and compare with non-relevant docs to find doc with highest rank which should be non-relevant
            # TODO: loop over actual top-k docs + scores

            # Get set of non-relevant documents for current query
            non_relevant_docs = self.relevance_data[(self.relevance_data.topic_id == query_id) & (self.relevance_data.relevancy == 0)]
            print(f"Amount of non-relevant docs for current query: {len(non_relevant_docs.cord_uid)}")

            # Check if one of the documents from the top-k documents are within this set
            for doc in ordered_doc_scores:
                non_relevant_top_k = non_relevant_docs[non_relevant_docs.cord_uid == doc]
                if np.array(non_relevant_top_k.cord_uid) != []:
                    # only get highest ranked document
                    break
            
            non_relevant_docs = non_relevant_top_k
            print(f"Non-relevant doc in the top-k docs: {non_relevant_docs}")
        else:
            non_relevant_docs = self.relevance_data[(self.relevance_data.topic_id == query_id) & (self.relevance_data.relevancy == 0)] # Only use positive feedback ?

        # Add terms in relevant & non-relevant docs to the collection term list
        # Create complete query vector accordingly.

        self.create_collection_list(relevant_docs.cord_uid)
        self.create_collection_list(non_relevant_docs.cord_uid)
        
        return [relevant_docs.cord_uid, non_relevant_docs.cord_uid]
    

    def rocchio_algorithm(self, qid, q0, top_docs, m):
        print("in rocchio algorithm")
        self.t.start()
        doc_ids = self.get_relevance_docs(qid, q0, m, top_docs)
        relevant_doc_ids = doc_ids[0]
        non_relevant_doc_ids = doc_ids[1]

        relevant_doc_vectors = []
        for doc_id in tqdm(relevant_doc_ids):
            relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
        print("Got relevant vectors")
        self.t.stop()

        self.t.start()
        non_relevant_doc_vectors = []
        for doc_id in tqdm(non_relevant_doc_ids):
            non_relevant_doc_vectors.append(self.create_complete_vector(self.tf_idf_docid(doc_id)))
        print("Got non-relevant vectors")
        self.t.stop()

        # Standard values
        alpha = 1.0
        beta = 0.75
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
        if non_relevant_doc_vectors == []:
            # no non-relevant feedback
            gamma = 0
            centroid_non_relevant_docs = 0
        else:
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
        return q_mod

    def rocchio_ranking(self, qid, q0, top_k_docs, model):
        rocchio_timer = Timer()
        rocchio_timer.start()
        if self.c_list == []:
            # start by adding query terms to the terms collection
            for term in q0:
                self.c_list.append(term)
            # add the terms from the top-k documents
            self.create_collection_list(top_k_docs)
        print("in rocchio ranking")

        q_mod = self.rocchio_algorithm(qid, q0, top_k_docs, model) # TODO: fix idh argument
        print("got qmod")

        # Check how many values are not 0 to check effect of relevance feedback 
        q_mod_diff = [x for x in q_mod if x != 0.0]
        print(f"q_mod non-zero terms: {len(q_mod_diff)}")

        doc_scores = {}

        count = 0

        self.t.start()

        # Rank documents using dot product as similarity function
        for doc in tqdm(top_k_docs):
            similarity_score = np.dot(np.array(list(self.create_complete_vector(self.tf_idf_docid(doc)))), q_mod)
            doc_scores[doc] = similarity_score
            #print(f"Doc nr.:{count} - Score: {similarity_score}")
            #count += 1
        self.t.stop()
    
        #print(doc_scores)
        print("TOTAL ROCCHIO TIME: ")
        rocchio_timer.stop()
        self.c_list = []
        return doc_scores

"""
# code to plot graph with the balance of relevancy per topic.
import plotly.express as px

fig = px.histogram(relevance_data, x="topic_id", color = "relevancy")
fig.show()
"""
