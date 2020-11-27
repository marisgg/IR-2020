def write_output(query_id,doc_id,ranking,score,name):
    with open("ranking.txt",'r') as r:
        for i in range(len(ranking)):
            r.write("{}\t{}\t{}\t{}\t  {score:.{6}f}\t{}".format(str(query_id), "Q0", str(doc_id), ranking, score,name))