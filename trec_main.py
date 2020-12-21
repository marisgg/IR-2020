#!/usr/bin/env python3
import argparse
import itertools
import json
import xml.etree.ElementTree as ET
import sys
import heapq
import os
import pickle
import time
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from progress.bar import Bar
import pytrec_eval
from output import write_output
from models import Models
from index_trec import Index, InvertedList

lucene_index = "lucene-index-cord19-abstract-2020-05-19"
qrelfile = "input/qrels-covid_d3_j0.5-3.txt"
topicsfile = "input/topics-rnd3.xml"

def dummy_document_at_a_time(query, index, models, k):
    L = []
    R = []
    for term in analyze_query(query):
        l = index.get_inverted_list(term)
        L.append(l)
    print(L)
    for d in range(192459):
        score = 0
        for l in L:
            print(l.get_current_doc() is None)
            if l.get_current_doc() == d:
                if verbose:
                    print("Computing score")
                score += models.bm25_term(index.get_docid_from_index(d), l.get_term())
                print(score)
            l.skip_forward_to_document(d)
            l.increment()
        if len(R) > k:
            heapq.heappushpop(R, (score, index.get_docid_from_index(d)))
            if verbose:
                print("Replaced heap")
        else:
            heapq.heappush(R, (score, index.get_docid_from_index(d)))
            if verbose:
                print("Pushed to heap")
    result = [heapq.heappop(R) for _ in range(min(k, len(R)))]
    if verbose:
        print(result)
    return result

def document_at_a_time(query, index, models, k, docidx_docid):
    L = []
    R = []
    for term in analyze_query(query):
        l = index.get_inverted_list(term)
        L.append(l)
    # Sort array of inverted lists by smallest list first
    L = sorted(L, key=lambda item: item.get_list_len())
    d = -1
    found = []
    longest_doc = None
    finished = False
    while(not finished):
        score = 0
        for l in L:
            doc = l.get_current_doc()
            if doc is None:
                continue
            doc_length = docidx_docid[doc][1]
            if doc_length > d:
                d = doc_length
                longest_doc = doc
        assert d != -1
        assert d == index.get_n_of_words_in_inverted_list_doc(longest_doc)
        for l in L:
            l.skip_forward_to_document(longest_doc)
            if l.get_current_doc() == longest_doc:
                # score += models.bm25_term(index.get_docid_from_index(longest_doc), l.get_term())
                docid = docidx_docid[longest_doc][0]
                score += models.bm25_term(docid, l.get_term())
                # score += models.tf_idf_term(docid, l.get_term())
                l.increment()
            else:
                d = -1
        if d > -1 and not docid in found:
            if len(R) < k:
                heapq.heappush(R, (score, docidx_docid[longest_doc][0]))
            else:
                heapq.heappushpop(R, (score, docidx_docid[longest_doc][0]))
            found.append(docid)
        finished = any([l.is_finished() for l in L])
    result = sorted([(score, doc_id) for score, doc_id in R], key=lambda item : item[0], reverse=True)
    if verbose:
        print(result)
    return result

def parse_topics(topicsfilename):
    topics = {}
    root = ET.parse(topicsfilename).getroot()
    for topic in root.findall("topic"):
        topic_number = topic.attrib["number"]
        topics[topic_number] = {}
        for query in topic.findall("query"):
            topics[topic_number]["query"] = query.text
        for question in topic.findall("question"):
            topics[topic_number]["question"] = question.text
        for narrative in topic.findall("narrative"):
            topics[topic_number]["narrative"] = narrative.text
    return topics

def write_topics_to_json(filename):
    topics = parse_topics(filename)
    with open("topics.json", "w+") as outfile:
        json.dump(topics, outfile)

def read_json_topics(filename):
    with open(filename, "r") as infile:
        topics = json.load(infile)
        return topics

def preprocess_query(query):
    stop_words = ["a","about","after","all","also","always","am","an","and","any","are","at","be","been","being","but","by","came","can","cant","come","could","did","didnt","do","does","doesnt","doing","dont","else","for","from","get","give","goes","going","had","happen","has","have","having","how","i","if","ill","im","in","into","is","isnt","it","its","ive","just","keep","let","like","made","make","many","may","me","mean","more","most","much","no","not","now","of","only","or","our","really","say","see","some","something","take","tell","than","that","the","their","them","then","there","they","thing","this","to","try","up","us","use","used","uses","very","want","was","way","we","what","when","where","which","who","why","will","with","without","wont","you","your","youre"]
    return [word for word in query.split() if word not in stop_words]

def score_query_heap(query, ranking_function, docs, index_class, models_class, k):
    R = []
    count = 0
    if verbose:
        print(query)
        bar = Bar("Computing scores for query", max=(len(docs)))
    for term in query:
        models_class.compute_df_vector(term)
    for doc in docs:
        score = ranking_function(models_class, doc, query)
        if verbose:
            bar.next()
        if len(R) < k:
            heapq.heappush(R, (score, doc))
        else:
            heapq.heappushpop(R, (score, doc))
    if verbose:
        bar.finish()
    # models_class.reset_df_vector()
    return sorted([(score, doc_id) for score, doc_id in R], key=lambda item : item[0], reverse=True)

def score_query(query, ranking_function, docs, index_class, models_class):
    doc_scores = {}
    count = 0
    if verbose:
        print(query)
        bar = Bar("Computing scores for query", max=(len(docs)))
    for term in query:
        models_class.compute_df_vector(term)
    for doc in docs:
        score = ranking_function(models_class, doc, query)
        if verbose:
            bar.next()
        if score > 0:
            doc_scores[doc] = score
    if verbose:
        bar.finish()
    models_class.reset_df_vector()
    return doc_scores

def score_tf_idf(m_class, doc, query):
    return m_class.tf_idf_query(doc, query)

def score_bm25(m_class, doc, query):
    return m_class.bm25_query_score(doc, query)

def analyze_query(query):
    analyzer = Analyzer(get_lucene_analyzer())
    query = analyzer.analyze(query)
    return query

def get_docs_and_score_query(query, ranking_function, index_class, models_class, topic_id, k, docidx_docid, rerank="none"):
    docs_list = []

    query = analyze_query(query)
    for term in query:
        docs = index_class.get_docids_from_postings(term, docidx_docid, debug=False)
        print(term)
        print(len(docs))
        docs_list.append(docs)
    docs = set(itertools.chain.from_iterable(docs_list))

    doc_scores = score_query_heap(query, ranking_function, docs, index_class, models_class, k)
    print(doc_scores)

    if rerank != "none":
        if verbose:
            print("Using {rerank} for reranking")
        top_k_docs = list(map(lambda x : x[1], doc_scores[:k]))
        doc_scores = models_class.rocchio_ranking(topic_id, query, top_k_docs, rerank)
        doc_scores = sorted([(v, k) for k, v in doc_scores.items()], key=lambda item : item[0], reverse=True)

    ## reranking of the ranked documents (Rocchio algorithm) ## top-k ?
    ## Assume that the top-k ranked documents are relevant. 

    # ordered_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:k])
    # return ordered_doc_scores

    return doc_scores

def pytrec_dictionary_entry(qid, docid, score):
    """ Create dictionary entry to update the total run output for pytrec_eval """
    return {
                'q' + str(qid) : {
                    str(docid) : score
                }
            }

def pytrec_evaluation(runfile, qrelfile, measures = pytrec_eval.supported_measures):
    """ run trec_eval with "measures" from the Python interface """
    with open(runfile, "r") as ranking:
        run = pytrec_eval.parse_run(ranking)
    with open(qrelfile, "r") as qrel:
        qrel = pytrec_eval.parse_qrel(qrel)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, measures)

    return evaluator.evaluate(run)

def cheat(filename, models, index, docidx_docid):
    cheat_dict = {}
    bar = Bar("Creating tf-idf dictionary", max=(index.get_max_docindex()))
    for doc in range(index.get_max_docindex()):
        try:
            docid = docidx_docid[doc][0]
            doclen = docidx_docid[doc][1]
            cheat_dict[doc] = models.tf_idf_docid(docid, doclen)
            bar.next()
        except:
            print("Error!")
            continue
    bar.finish()
    with open(filename, 'wb') as handle:
        pickle.dump(cheat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped dictionary")

def run(k1=0.9, b=0.4):
    parser = argparse.ArgumentParser(description="TREC-COVID document ranker CLI")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", default=False)
    parser.add_argument("-cp", "--compute_pickle", action="store_true", default=False)
    parser.add_argument("-n", "--n_queries", help="maximum number of queries to run", type=int, default=999)
    parser.add_argument("-m", "--model", help="which model used in ranking", default="bm25")
    parser.add_argument("-d", "--doc_at_a_time", help="Use document_at_a_time algorithm", action="store_true", default=False)
    parser.add_argument("-k", "--k_docs", help="Numer of documents to retrieve", type=int, default=100)
    parser.add_argument("-r", "--rerank", help="which rerank model to use: 'none', 'rocchio', or 'ide'", default="none")
    global k1_param
    global b_param
    k1_param = k1
    b_param = b
    args = parser.parse_args()
    global verbose
    verbose = args.verbose
    model = args.model
    doc_at_a_time = args.doc_at_a_time
    k = args.k_docs
    rerank = args.rerank

    index_reader = IndexReader(lucene_index)
    searcher = SimpleSearcher(lucene_index)
    models = Models(index_reader, qrelfile)
    trec_index = Index(index_reader, searcher)

    with open("tf_idf.pickle", 'rb') as handle:
        dtfidf = pickle.load(handle)

    if not os.path.exists('output'):
        os.makedirs('output')

    if args.compute_pickle:
        print("Computing id index dict")
        docidx_docid = {docidx : (trec_index.get_docid_from_index(docidx), trec_index.get_n_of_words_in_inverted_list_doc(docidx)) for docidx in range(trec_index.get_max_docindex())}
        with open('filename.pickle', 'wb') as handle:
            pickle.dump(docidx_docid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if True:
        with open('filename.pickle', 'rb') as handle:
            print("Loading id index dict")
            docidx_docid = pickle.load(handle)
        print("Finished initializing id index dict")

        models.set_docid_tf_idf(dtfidf, {v[0]: k for k, v in docidx_docid.items()})
        dtfidf = None

    topics = parse_topics(topicsfile)

    rocchio = False
    if model == "bm25":
        rankfun = score_bm25
    elif model == "tf_idf":
        rankfun = score_tf_idf
    else:
        print("Model should be 'tf_idf' or 'bm25' (default)!")
        sys.exit(1)

    t = time.localtime()
    current_time = time.strftime("%H.%M", t)
    formatstring = "output/benchmark/3-ranking-{0}-{1}-{2}"
    rankfile = formatstring.format(model, current_time, rerank) + ".txt"
    resultfile = formatstring.format(model, current_time, rerank) + ".json"

    if doc_at_a_time:
        try:
            with open(rankfile, 'w') as outfile:
                for idx in range(1, min(args.n_queries+1, len(topics)+1)):
                    for i, (score, docid) in enumerate(document_at_a_time(topics[str(idx)]["query"], trec_index, models, k, docidx_docid), 1):
                        outfile.write(write_output(idx, docid, i, score, "document_at_a_time"))
        finally:
            outfile.close()
    else:
        try:
            with open(rankfile, 'w') as outfile:
                for idx in range(1, min(args.n_queries+1, len(topics)+1)):
                    for i, (score, docid) in enumerate(
                        get_docs_and_score_query(topics[str(idx)]["query"], rankfun, trec_index, models, idx, k, docidx_docid, rerank=rerank), 1):
                        outfile.write(write_output(idx, docid, i, score, "score_query"))
        finally:
            outfile.close()

    results = pytrec_evaluation(rankfile, qrelfile)
    with open(resultfile, 'w') as outjson:
        json.dump(results, outjson)

def main():
    run()

if __name__ == "__main__":
    main()
