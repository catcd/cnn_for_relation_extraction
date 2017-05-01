import numpy as np

WORD_EMBEDDING_DIR = "data/word_vec/"
F50DT = "model50.directed.tag.vec"
F50DV = "model50.directed.value.vec"
F50UT = "model50.undirected.tag.vec"
F50UV = "model50.undirected.value.vec"
F50NT = "model50.noedge.tag.vec"
F50NV = "model50.noedge.value.vec"
F100DT = "model100.directed.tag.vec"
F100DV = "model100.directed.value.vec"
F100UT = "model100.undirected.tag.vec"
F100UV = "model100.undirected.value.vec"
F100NT = "model100.noedge.tag.vec"
F100NV = "model100.noedge.value.vec"
F300DT = "model300.directed.tag.vec"
F300DV = "model300.directed.value.vec"
F300UT = "model300.undirected.tag.vec"
F300UV = "model300.undirected.value.vec"
F300NT = "model300.noedge.tag.vec"
F300NV = "model300.noedge.value.vec"

EMBED_SIZE = {
    F50DT:   50, F50DV:   50, F50UT:   50, F50UV:   50, F50NT:   50, F50NV:   50,
    F100DT: 100, F100DV: 100, F100UT: 100, F100UV: 100, F100NT: 100, F100NV: 100,
    F300DT: 300, F300DV: 300, F300UT: 300, F300UV: 300, F300NT: 300, F300NV: 300
}

SENTENCE_LENGTH = 60

word_map = {}


def sentence2matrix(we_key, sentence):
    s = sentence.split()
    if len(s) > SENTENCE_LENGTH:
        started = (len(s)-SENTENCE_LENGTH)/2
        s = s[started:started + SENTENCE_LENGTH]

    ret = np.zeros((SENTENCE_LENGTH, EMBED_SIZE.get(we_key, 0)))

    for i in range(len(s)):
        ret[i] = word2vec(we_key, s[i])

    return ret


def word2vec(we_key, word):
    if word_map.get(we_key) is None:
        __load_word_map(we_key, WORD_EMBEDDING_DIR + we_key)

    if word in word_map.get(we_key):
        return word_map.get(we_key)[word]
    else:
        return np.zeros(EMBED_SIZE.get(we_key, 0))


def __load_word_map(we_key, we_file):
    print("Load " + we_key)
    global word_map
    word_map[we_key] = {}

    # Load data from files
    word_data = list(open(we_file, "r").readlines())

    # Generate labels
    for word in word_data:
        t = word.split()
        word_map[we_key][t[0]] = np.array(list(map(float, t[1:])))

    print("Load finished")
