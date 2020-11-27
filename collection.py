from pyserini.collection import pycollection

collection = pycollection.Collection('Cord19AbstractCollection', 'C:\Users\noekn\Documents\Information_Retrieval')

cnt = 0;
full_text = {True : 0, False: 0}

articles = collection.next()
for (i, d) in enumerate(articles):
    article = pycollection.Cord19Article(d.raw)
    cnt = cnt + 1
    full_text[article.is_full_text()] += 1
    if cnt % 1000 == 0:
        print(f'{cnt} articles read...')