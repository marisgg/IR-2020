#!/usr/bin/env python3
import argparse
import itertools
import json
import xml.etree.ElementTree as ET
import sys
import heapq
from multiprocessing import Pool
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from progress.bar import Bar
import pytrec_eval
from output import write_output
from models import Models
from index_trec import Index, InvertedList
import pickle

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
    docidx_docid = {}


    d = -1
    longest_doc = None
    finished = False
    while(not finished):
        score = 0
        for l in L:
            doc = l.get_current_doc()
            if doc is None:
                continue
            type(doc)
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
                score += models.bm25_term(docidx_docid[longest_doc][0], l.get_term())
                l.increment()
            else:
                d = -1
        if d > -1:
            if len(R) < k:
                heapq.heappush(R, (score, longest_doc))
            else:
                heapq.heappushpop(R, (score, longest_doc))
        finished = any([l.is_finished() for l in L])
    result = sorted(list(set([heapq.heappop(R) for _ in range(min(k, len(R)))])), key=lambda item : item[0], reverse=True)
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

def score_query(query, model, docs, index_class, models_class):
    doc_scores = {}
    count = 0
    if verbose:
        print(query)
        bar = Bar("Computing scores for query", max=(len(docs)))
    if model == "bm25":
        for doc in docs:
            score = models_class.bm25_query_score(doc, query)
            if verbose:
                bar.next()
            if score > 0:
                doc_scores[doc] = score
    elif model == "tf_idf":
        for doc in docs:
            score = 0
            for term in query:
                score += models_class.tf_idf_term(term, doc)
                count += 1
                if verbose and count % 1000 == 0:
                    bar.next()
            if score > 0:
                doc_scores[doc] = score
    if verbose:
        bar.finish()   
    return doc_scores

def score_bm25():
    pass

def score_bm25_query(query, model, docs, models_class):
    doc_scores = {}
    for doc in docs:
        score = models_class.bm25_docid_query(doc, query)
        if score > 0.0:
            doc_scores[doc] = score
    return doc_scores

def analyze_query(query):
    analyzer = Analyzer(get_lucene_analyzer())
    query = analyzer.analyze(query)
    return query

def get_docs_and_score_query(query, model, index_class, models_class):
    docs = set()

    # TODO: Get documents in which percentage of query terms exist? 
    """
    Ik stel het volgende voor (zonder onderbouwing verder): als query > 3 woorden bevat, kijken we
    naar documenten waarin minimaal 2 woorden zitten? (50%)
    --> Iig moeten we hier iets voor bedenken denk ik.
    """
    for term in query:
        docs = index_class.get_docids_from_postings(term, return_set = docs, debug=True)
        if verbose:
            print(term)
            print(len(docs))

    if False:
        doc_scores = score_bm25_query(query, model, docs, models_class)
    else:
        doc_scores = score_query(query, model, docs, index_class, models_class)

    # TODO: Take the top 1000 for output writing
    ordered_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:10000])
    return ordered_doc_scores

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

def main():
    parser = argparse.ArgumentParser(description="TREC-COVID document ranker CLI")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", default=False)
    parser.add_argument("-cp", "--compute_pickle", action="store_true", default=False)
    parser.add_argument('-query', default="covid symptoms")
    parser.add_argument("-j", "--json", help="generate json from topics list", action="store_true", default=False)
    parser.add_argument("-n", "--n_queries", help="maximum number of queries to run", type=int, default=999)
    parser.add_argument("-m", "--model", help="which model used in ranking", default="bm25")
    args = parser.parse_args()
    query = args.query
    global verbose
    verbose = args.verbose
    model = args.model

    index_reader = IndexReader('lucene-index-cord19-abstract-2020-07-16')
    searcher = SimpleSearcher('lucene-index-cord19-abstract-2020-07-16')
    models = Models(index_reader, searcher)
    trec_index = Index(index_reader, searcher)

    if args.compute_pickle:
        print("Computing id index dict")
        docidx_docid = {docidx : (trec_index.get_docid_from_index(docidx), trec_index.get_n_of_words_in_inverted_list_doc(docidx)) for docidx in range(trec_index.get_max_docindex())}
        with open('filename.pickle', 'wb') as handle:
            pickle.dump(docidx_docid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('filename.pickle', 'rb') as handle:
        print("Loading id index dict")
        docidx_docid = pickle.load(handle)
        print(docidx_docid[int(str(14))])
    print("Finished initializing id index dict")

    if args.json:
        # only need to do this once, program small MD5 or something
        write_topics_to_json("topics-rnd5.xml")

    topics = read_json_topics("topics.json")
    # print(document_at_a_time(topics[str(1)]["query"], trec_index, models, 10))
    try:
        with open("ranking.txt", 'w') as outfile:
            for idx in range(1, min(args.n_queries+1, len(topics)+1)):
                for i, (score, docid) in enumerate(document_at_a_time(topics[str(idx)]["query"], trec_index, models, 100, docidx_docid), 1):
                    outfile.write(write_output(idx, trec_index.get_docid_from_index(docid), i, score, "testrun"))
    finally:
        outfile.close()

    results = pytrec_evaluation("ranking.txt", "qrels-covid_d5_j0.5-5.txt")
    with open("results.json", 'w') as outjson:
        json.dump(results, outjson)

if __name__ == "__main__":
    main()
