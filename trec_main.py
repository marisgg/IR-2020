#!/usr/bin/env python3
import argparse
from timeit import Timer
from process_data import process_data
from preprocess_data_optimized import preprocessing
import index_trec
from output import write_output
import xml.etree.ElementTree as ET

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

def score_query(query, docids=None):
    doc_scores = {}
    docs = []
    scores = []

    # TODO: Get documents in which percentage of query terms exist? 
    """
    Ik stel het volgende voor (zonder onderbouwing verder): als query > 3 woorden bevat, kijken we
    naar documenten waarin minimaal 2 woorden zitten? (50%)
    --> Iig moeten we hier iets voor bedenken denk ik.
    """

    for term in query.split():
        for doc in index_trec.get_docids(term):
            docs.append(doc)

    for doc in docs:
        score = 0
        for term in query.split():
            score += index_trec.tf_idf_term(term, doc)
        doc_scores[doc] = score

    ordered_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1]), reverse=True)
    return ordered_doc_scores


def main():
    parser = argparse.ArgumentParser(description="TREC-COVID document ranker CLI")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", default=True)
    parser.add_argument('-query', default="covid symptoms")
    args = parser.parse_args()
    query = args.query

    topics = parse_topics("topics-rnd5.xml")

    for idx in range(1, 50):
        for docid, score in score_query(topics[str(idx)]["query"]).items():
            if(docid == "reverse"):
                continue
            write_output(idx, docid, -1, score, topics[str(idx)]["query"])
  
if __name__ == "__main__":
    main()
