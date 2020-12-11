import itertools
import math

class Index:

    def __init__(self, index, searcher):
        self.index_reader = index
        self.searcher = searcher

    def get_docids(self, term, max_doc=192459) -> []:
        """ gets ALL docids by default order until the max_doc limit (defaults to num_docs) """
        return [self.searcher.doc(i).docid() for i in range(self.searcher.num_docs)]

    def get_docids_from_postings(self, term, max_doc=192459, debug=False) -> set():
        """ Use postings and set union to get list of documents containing query words """
        if debug:
            return_set = set()
            for posting in self.index_reader.get_postings_list(term):
                try:
                    docnum = posting.docid
                    doc = self.searcher.doc(docnum)
                    docid = doc.docid()
                    return_set |= docid
                except:
                    continue
        return set([self.searcher.doc(posting.docid).docid() for posting in self.index_reader.get_postings_list(term) if posting.docid ])

    def term_in_doc(self, term, docid) -> bool:
        """ Don't use in production, overly complex """
        return docid in self.searcher.search(term)

    def get_docids_with_search(self, term, max_doc=10) -> []:
        # First find docids within the prebuild index
        hits = self.searcher.search(term)
        hits_docid = []
        for i in range(min(len(hits), max_doc)):
            hits_docid.append(hits[i].docid)
        return hits_docid
