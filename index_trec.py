from pyserini import analysis, index
import itertools
from pyserini.search import SimpleSearcher
import math

index_reader = index.IndexReader('lucene-index-cord19-abstract-2020-07-16')

# Access basic index statistics
print(index_reader.stats())
total_documents = index_reader.stats()['documents']
print(f'total amount of documents: {total_documents}')

#Iterate over index terms and access term statistics
for term in itertools.islice(index_reader.terms(), 10):
    print(f'{term.term} (df={term.df}, cf={term.cf})')

term = 'cities'

# Analyze the term.
analyzed = index_reader.analyze(term)
print(f'The analyzed form of "{term}" is "{analyzed[0]}"')

# Skip term analysis:
df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)
print(f'term "{term}": df={df}, cf={cf}')

# Fetch and traverse postings for an unanalyzed term
postings_list = index_reader.get_postings_list(term)
for posting in postings_list:
    #print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')
    pass

# Fetch and traverse postings for an analyzed term:
postings_list = index_reader.get_postings_list(analyzed[0], analyzer=None)
for posting in postings_list:
    #print(f'docid={posting.docid}, tf={posting.tf}, pos={posting.positions}')
    pass


# First find docids within the prebuild index
searcher = SimpleSearcher('lucene-index-cord19-abstract-2020-07-16')
hits = searcher.search('lung')
# Print the first 10 hits:
#for i in range(0, 10):
#    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')
test_docid = hits[0].docid
print(f'test docid: {test_docid}')

# Fetch the document vector for a document
doc_vector = index_reader.get_document_vector(test_docid)
#print(f'Document vector: {doc_vector}') # dict where keys are the analyzed terms and values the term frequenies


# Compute the tf-idf representation
tf = index_reader.get_document_vector(test_docid)
df = {term: (index_reader.get_term_counts(term, analyzer=None))[0] for term in tf.keys()} 

def get_docids(term, max_doc=10):
    # First find docids within the prebuild index
    searcher = SimpleSearcher('lucene-index-cord19-abstract-2020-07-16')
    hits = searcher.search(term)
    hits_docid = []
    for i in range(max_doc):
        hits_docid.append(hits[i].docid)

    return hits_docid

# print(f'tf: {tf}')
# print(f'df: {df}')


N = total_documents = index_reader.stats()['documents']
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
def tf_idf_term(term, docid):
    # TODO: tf should be divided by number of words in docid
    tfs = index_reader.get_document_vector(docid)
    if term in tfs:
        tf = tfs[term]
        df = index_reader.get_term_counts(term, analyzer=None)[0]
        return tf * math.log(N / (df + 1))
    else: 
        return 0

def tf_idf_docid(docid):
    doc_vector = index_reader.get_document_vector(docid)
    tf_idf = {}
    for term, count in doc_vector.items():
        df = index_reader.get_term_counts(term, analyzer=None)[0]
        tf_idf[term] = count * math.log(N / (df + 1))
    return tf_idf

def tf_idf():
    pass
