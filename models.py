from pyserini import analysis, index
import itertools
from pyserini.search import SimpleSearcher
import math

index_reader = index.IndexReader('lucene-index-cord19-abstract-2020-07-16')
searcher = SimpleSearcher('lucene-index-cord19-abstract-2020-07-16')

"""
Compute TF-IDF, which consists of the following two components:
1. Term frequency: measures the frequency of a word in a document, normalize.
    tf(t,d) = count of t in d / number of words in d
2. Inverse document frequency: measures the informativeness of term t.
    idf(t) = log(N / (df + 1)               (df = occurences of t in documents)

The resulting formula: tf-idf(t,d) = tf(t,d)*log(N/(df+1))

INPUT:		Dictionary, with for each file a sub-dictionary containing the title, abstract, and introduction.
OUTPUT:		
"""

""" Hacky: Sum all term frequencies in document vector (thus no stopwords) """
def get_n_of_words_in_docid(docid):
    return sum(index_reader.get_document_vector(docid).values())


def tf_idf_term(term, docid):
    N = index_reader.stats()['documents']
    tfs = index_reader.get_document_vector(docid)
    if term in tfs:
        tf = tfs[term]/get_n_of_words_in_docid(docid)
        df = index_reader.get_term_counts(term, analyzer=None)[0]
        return tf * math.log(N / (df + 1))
    else: 
        return 0


def tf_idf_docid(docid):
    N = index_reader.stats()['documents']

    tfs = index_reader.get_document_vector(docid)
    tf_idf = {}
    for term, count in tfs.items():
        df = index_reader.get_term_counts(term, analyzer=None)[0]
        tf_idf[term] = count/get_n_of_words_in_docid(docid) * math.log(N / (df + 1)) # added total number of words in doc
    return tf_idf

def bm25_term(term, docid):
    return index_reader.compute_bm25_term_weight(docid, term, analyzer=None)
     
def bm25_docid(docid):
    # get all terms in documents
    tfs = index_reader.get_document_vector(docid)
    bm25_vector = {term: index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in tfs.keys()}
    return bm25_vector


'''hits = searcher.search('corona')
# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')
test_docid = hits[0].docid
print(f'test docid: {test_docid}')'''
