def clear_output(filename):
    ''' clear output file '''
    open(filename, "w").close()

def write_output(query_id, doc_id, ranking, score, name):
    ''' write output that can be parsed by the trec_eval evaluation engine. Used to compare our rankings to the human query relevance rankings for the CORD-19 set '''
    out = "{}\t{}\t{}\t{}\t  {:.6f}\t{}\n".format(str(query_id), "Q0", str(doc_id), ranking, score, name)
    return out