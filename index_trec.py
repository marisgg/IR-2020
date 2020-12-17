import itertools
import math

class InvertedList:
    def __init__(self, pointer, term, ilist):
        self.pointer = pointer
        self.term = term
        self.ilist = ilist

    def get_item(self):
        # If list is either empty or finished
        if len(self.ilist) == 0 or self.is_finished:
            return None
        return self.ilist[self.pointer]

    def increment(self):
        self.pointer += 1

    def is_finished(self):
        return self.pointer == len(self.ilist)

    def get_current_doc(self):
        if self.get_item() is None:
            return None
        else:
            return self.get_item()[0]

    def get_current_tf(self):
        if self.get_item() is None:
            return None
        else:
            return self.get_item()[1]

    def skip_forward_to_document(self, docidx):
        while(not self.is_finished and self.get_current_doc != docidx):
            self.increment()

class Index:

    def __init__(self, index, searcher):
        self.index_reader = index
        self.searcher = searcher

    def get_docids(self, term, max_doc=192459) -> []:
        """ gets ALL docids by default order until the max_doc limit (defaults to num_docs) """
        return [self.searcher.doc(i).docid() for i in range(self.searcher.num_docs)]

    def get_docid_from_index(self, idx):
        return self.searcher.doc(idx).docid()

    def get_n_of_words_in_inverted_list_doc(self, doc):
        """ Hacky: Sum all term frequencies in document vector (thus no stopwords) """
        return sum(self.index_reader.get_document_vector(self.get_docid_from_index(doc)).values())

    def get_inverted_list(self, term):
        print(term)
        postings = self.index_reader.get_postings_list(term, analyzer=None)
        if postings is None:
            return InvertedList(0, term, [])
        else:
            # [(term, (index_docid, tf))]
            return InvertedList(0, term, sorted([(posting.docid, posting.tf) for posting in postings], key=lambda item: item[0]))
            # return [(term, (posting.docid, posting.tf)) for posting in postings]


    def get_docids_from_postings(self, term, return_set = set(), max_doc=192459, debug=False) -> set():
        """ Use postings and set union to get list of documents containing query words """
        if debug:
            try:
                postings = self.index_reader.get_postings_list(term)
            except:
                postings = []
            if postings != None:
                for posting in postings:
                    try:
                        docnum = posting.docid
                        doc = self.searcher.doc(docnum)
                        docid = doc.docid()
                        return_set |= set([docid])
                    except:
                        continue
            return return_set
        return [self.searcher.doc(posting.docid).docid() for posting in self.index_reader.get_postings_list(term) if posting is not None]

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
