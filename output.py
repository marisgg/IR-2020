def write_output(query_id,doc_id,ranking,score,name):
    with open("ranking.txt",'a') as r:
        out = "{}\t{}\t{}\t{}\t  {:.6f}\t{}\n".format(str(query_id), "Q0", str(doc_id), ranking, score, name)
        r.write(out)