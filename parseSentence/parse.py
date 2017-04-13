import os

from nltk.tokenize import sent_tokenize as st

chMap = {}
deMap = {}


def make_chemical_map():
    l = ['CTD_chemicals', 'chemical']
    for fi in l:
        _fou = open(fi, 'r')
        lines = _fou.readlines()
        for line in lines:
            ws = line.lower().split()
            data = True
            for w in ws[:0:-1]:
                data = {w: data}
            chMap[ws[0]] = data


def make_disease_map():
    l = ['CTD_diseases', 'disease']
    for fi in l:
        _fou = open(fi, 'r')
        lines = _fou.readlines()
        for line in lines:
            ws = line.lower().split()
            data = True
            for w in ws[:0:-1]:
                data = {w: data}
            deMap[ws[0]] = data


def match_chemical(_sent):
    """
    :type _sent: str
    """
    _ws = _sent.lower().split()

    for _i in xrange(len(_ws)):
        d = chMap
        j = _i
        while j < len(_ws):
            d = d.get(_ws[j])
            if True == d:
                return True
            elif d is None:
                break
            else:
                j += 1

    return False


def match_disease(_sent):
    """
    :type _sent: str
    """
    _ws = _sent.lower().split()

    for _i in xrange(len(_ws)):
        d = deMap
        j = _i
        while j < len(_ws):
            d = d.get(_ws[j])
            if True == d:
                return True
            elif d is None:
                break
            else:
                j += 1

    return False

make_chemical_map()
make_disease_map()

fou = open('/home/catcan/Desktop/data/dict.sent', 'w')
ec = 0
for fname in os.listdir('/home/catcan/GitProjects/word_embedding_trainer/data/pubmed_text_only'):
    try:
        fin = open('/home/catcan/GitProjects/word_embedding_trainer/data/pubmed_text_only/' + fname, 'r')
        abstracts = fin.readlines()
        for abt in abstracts:
            sents = st(abt.decode('utf-8'))
            for sent in sents:
                if match_chemical(sent) and match_disease(sent):
                    fou.write((sent + '\n').encode('utf-8'))
    except Exception as e:
        print(fname)
        print(e)
        ec += 1

print(ec)
