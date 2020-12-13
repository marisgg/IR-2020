#!/usr/bin/env python3
import argparse
import itertools
import json
import xml.etree.ElementTree as ET
import sys
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from output import write_output
from models import Models
from index_trec import Index
from progress.bar import Bar
import pytrec_eval

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

def score_query(query, model, index_class, models_class):
    doc_scores = {}
    docs = set()
    query = preprocess_query(query)
    # TODO: Get documents in which percentage of query terms exist? 
    """
    Ik stel het volgende voor (zonder onderbouwing verder): als query > 3 woorden bevat, kijken we
    naar documenten waarin minimaal 2 woorden zitten? (50%)
    --> Iig moeten we hier iets voor bedenken denk ik.
    """
    for term in query:
        postings = index_class.get_docids_from_postings(term, debug=False)
        docs |= postings
        if verbose:
            print(term)
            print(len(docs))

    count = 0
    if verbose:
        print(query)
        bar = Bar("Computing scores for query", max=(len(list(docs))*len(query))/1000)
    for doc in list(docs):
        score = 0
        for term in query:
            if model == "bm25":
                score += models_class.bm25_term(term, doc)
            elif model == "tf_idf":
                score += models_class.tf_idf_term(term, doc)
            else:
                print("No model found")
                sys.exit(1)
            count += 1
            if verbose and count % 1000 == 0:
                bar.next()
        if score > 0:
            doc_scores[doc] = score
    if verbose:
        bar.finish()
    # TODO: Take the top 1000 for output writing
    ordered_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1]), reverse=True)
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
    parser.add_argument('-query', default="covid symptoms")
    parser.add_argument("-j", "--json", help="generate json from topics list", action="store_true", default=False)
    parser.add_argument("-n", "--n_queries", help="maximum number of queries to run", type=int, default=1)
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

    if args.json:
        # only need to do this once, program small MD5 or something
        write_topics_to_json("topics-rnd5.xml")

    topics = read_json_topics("topics.json")
    try:
        with open("ranking.txt", 'w') as outfile:
            for idx in range(1, min(args.n_queries+1, 50)):
                for docid, score in score_query(topics[str(idx)]["query"], model, trec_index, models).items():
                    if docid == "reverse":
                        continue
                    outfile.write(write_output(idx, docid, -1, score, "testrun"))
    finally:
        outfile.close()

    results = pytrec_evaluation("ranking.txt", "qrels-covid_d5_j0.5-5.txt")
    print(results)

if __name__ == "__main__":
    main()
