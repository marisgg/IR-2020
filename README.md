# TREC-COVID Information Retrieval System
## Information Retrieval System for CORD-19 datasets for the course Information Retrieval at Radboud University

This repository consists of the code-base for our custom information retrieval system oriented towards the CORD-19 TREC-COVID dataset. It uses Pyserini for the inverted index and pytrec_eval for evaluation.

# Prerequisites

In order to run the system, you need a pre-built Lucene index, qrelfile, and topics file for the collection. You can get these as follows:

This code snippets gathers files for challenge round five. These files are expected to be available by default.
```bash
wget https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml --directory-prefix=input/
wget https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt --directory-prefix=input/
wget https://www.dropbox.com/s/9hfowxi7zenuaay/lucene-index-cord19-abstract-2020-07-16.tar.gz \
	&& tar -xf lucene-index-cord19-abstract-2020-07-16.tar.gz \
	&& rm -f lucene-index-cord19-abstract-2020-07-16.tar.gz
```

## Other Rounds
To gather files for round three:
```bash
wget https://ir.nist.gov/covidSubmit/data/topics-rnd3.xml --directory-prefix=input/
wget https://ir.nist.gov/covidSubmit/data/qrels-covid_d3_j0.5-3.txt --directory-prefix=input/
wget https://www.dropbox.com/s/7bbz6pm4rduqvx3/lucene-index-cord19-abstract-2020-05-19.tar.gz \
	&& tar -xf lucene-index-cord19-abstract-2020-05-19.tar.gz \
	&& rm -f lucene-index-cord19-abstract-2020-05-19.tar.gz
```

If using other index and topic/qrel files, you need to change the following global variables in trec_main.py to use the accompanying dataset files (round three in this case):

```python
LUCENE_INDEX = "lucene-index-cord19-abstract-2020-05-19"
QRELFILE = "input/qrels-covid_d3_j0.5-3.txt"
TOPICSFILE = "input/topics-rnd3.xml"
```

## Python package requirements

The Python program uses 

Install requirements:
```bash
python3 -m pip install -r requirements.txt
```

# Executing run

Then, you can run the program, see below for an example running five queries using the BM_25 ranking function without reranking:

```bash
python3 trec_main.py -v -n 5
```

# Results

Ranking results can be found in ```output/ranking-*.txt```, the trec_eval evaluation results can be found in ```output/results-*.txt```

## Usage of main file

```
usage: trec_main.py [-h] [-v] [-cp] [-n N_QUERIES] [-m MODEL] [-d] [-k K_DOCS] [-r RERANK]

TREC-COVID document ranker CLI

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  -cp, --compute_pickle
                        Compute mapping from internal lucene id's to external docid's
  -n N_QUERIES, --n_queries N_QUERIES
                        Maximum number of queries to run
  -m MODEL, --model MODEL
                        which model used in ranking from {bm25, tf_idf}
  -d, --doc_at_a_time   Use document_at_a_time algorithm
  -k K_DOCS, --k_docs K_DOCS
                        Numer of documents to retrieve
  -r RERANK, --rerank RERANK
                        Which rerank model to use 'rocchio', or 'ide'
```