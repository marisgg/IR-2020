#!/usr/bin/env python3
import argparse
from timeit import Timer
from process_data import process_data
from preprocess_data_optimized import preprocessing, preprocess_query
import index_trec
import output
import json
import xml.etree.ElementTree as ET
import numpy as np
import models

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

def score_query(query, model):
    doc_scores = {}
    docs = set()
    scores = []
    query = preprocess_query(query)
    # TODO: Get documents in which percentage of query terms exist? 
    """
    Ik stel het volgende voor (zonder onderbouwing verder): als query > 3 woorden bevat, kijken we
    naar documenten waarin minimaal 2 woorden zitten? (50%)
    --> Iig moeten we hier iets voor bedenken denk ik.
    """
    for term in query:
        postings = index_trec.get_docids_from_postings(term)
        docs |= postings
        if(verbose):
            print(term)
            print(len(docs))
    
    count = 0
    for doc in list(docs):
        score = 0
        for term in query:
            #score += index_trec.tf_idf_term(term, doc)
            if model == "bm25":
                score += models.bm25_term(term, doc)
            elif model == "tf_idf":
                score += models.tf_idf_term(term, doc)
            else:
                print("No model found")
            count += 1
            if(verbose and count % 1000 == 0):
                print("Processed {0} scores out of {1}..".format(count, len(list(docs))*len(query)))
        doc_scores[doc] = score

    ordered_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1]), reverse=True)
    return ordered_doc_scores


def main():
    parser = argparse.ArgumentParser(description="TREC-COVID document ranker CLI")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", default=False)
    parser.add_argument('-query', default="covid symptoms")
    parser.add_argument("-j", "--json", help="generate json from topics list", action="store_true", default=False)
    parser.add_argument("-n", "--n_queries", help="maximum number of queries to run", default=1)
    parser.add_argument("-m", "--model", help="which model used in ranking", default="bm25")
    args = parser.parse_args()
    query = args.query
    global verbose
    verbose = args.verbose
    model = args.model

    if args.json:
        # only need to do this once, program small MD5 or something
        write_topics_to_json("topics-rnd5.xml")

    topics = read_json_topics("topics.json")

    output.clear_output()
    for idx in range(1, min(args.n_queries+1, 50)):
        for docid, score in score_query(topics[str(idx)]["query"], model).items():
            if(docid == "reverse"):
                continue
            output.write_output(idx, docid, -1, score, topics[str(idx)]["query"])
  
if __name__ == "__main__":
    main()
