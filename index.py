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

print(f'tf: {tf}')
print(f'df: {df}')

# --> Combine to get tf-idf score
# TO DO
tf_idf = {}
for key, value in tf.items():
    idf = math.log(total_documents / df[key])
    tf_idf[key] = tf[key]*idf

print(f'tf-idf dictionary: \n {tf_idf}')

# Note that the keys of get_document_vector() are already analyzed, we set analyzer to be None.
#bm25_score = index_reader.compute_bm25_term_weight(test_docid, 'lung', analyzer=None)
#print(f'bm25 score for "lung" in {test_docid}: {bm25_score})

# Alternatively, we pass in the unanalyzed term:
bm25_score = index_reader.compute_bm25_term_weight(test_docid, 'lung')
print(f'bm25 score for "lung" in {test_docid}: {bm25_score}')

# Compute the score of a specific document w.r.t. a query
query = 'covid-19 symptoms'
for i in range(10):
    score = index_reader.compute_query_document_score(hits[i].docid, query)
    print(f'{i+1:2} {hits[i].docid:15} {score:.5f}')





'''# Get to know the positions of each term in the document
# function get_term_positions('XXXX') is not an attribute apparently?

term_positions = index_reader.get_term_position('FBIS4-67701')
print(term_positions)

# If you want to reconstruct the document using the position information
doc = []
for term, positions in term_positions.items():
    for p in positions:
        doc.append((term,p))

doc = ' '.join([t for t, p in sorted(doc, key=lambda x: x[1])])
print(doc)'''