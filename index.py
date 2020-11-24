from pyserini import analysis, index
import itertools

index_reader = index.IndexReader('lucene-index-cord19-abstract-2020-07-16')

for term in itertools.islice(index_reader.terms(), 0):
    print(f'{term.term} (df={term.df}, cf={term.cf})')

term = 'respiratori'

# Analyze the term.
analyzed = index_reader.analyze(term)
print(f'The analyzed form of "{term}" is "{analyzed[0]}"')

# Skip term analysis:
df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)
print(f'term "{term}": df={df}, cf={cf}')