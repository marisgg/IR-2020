''' clear output file '''
def clear_output():
	open("ranking.txt", "w").close()


''' write output that can be parsed by the trec_eval evaluation engine. Used to compare our rankings to the human query relevance rankings for the CORD-19 set '''
def write_output(query_id,doc_id,ranking,score,name):
    with open("ranking.txt",'a') as r:
        out = "{}\t{}\t{}\t{}\t  {:.6f}\t{}\n".format(str(query_id), "Q0", str(doc_id), ranking, score, name)
        r.write(out)