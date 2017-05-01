import numpy as np
from helpers import data_helpers as dh, we_helpers as wh

from models.model_s2345_f24_h1_32 import ModelS2345F24H132

np.random.seed(20)
EMBED_SIZE = 100
DATA_POS = "./data/training/full.directed.value.pos"
DATA_NEG = "./data/training/full.directed.value.neg"

TEST_POS = "./data/training/test.directed.value.pos"
TEST_NEG = "./data/training/test.directed.value.neg"


def do_h132():
    print("Loading data")
    x_data, y_data = dh.load_data_and_labels(DATA_POS, DATA_NEG)
    for i in range(len(x_data)):
        x_data[i] = wh.sentence2matrix(wh.F100DV, x_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print("Loaded")

    print("Loading test")
    x_test, y_test = dh.load_data_and_labels(TEST_POS, TEST_NEG)
    for i in range(len(x_test)):
        x_test[i] = wh.sentence2matrix(wh.F100DV, x_test[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("Loaded")

    print("Evaluate")
    model = ModelS2345F24H132(EMBED_SIZE, wh.SENTENCE_LENGTH)
    cvr = model.cross_validation(x_data, y_data)
    pm = np.mean(cvr, axis=0)
    print(cvr)
    print("p={} r={} f1={}".format(pm[0], pm[1], pm[2]))
    ter = model.train_separated_test(x_data, y_data, x_test, y_test)
    print(ter)


do_h132()
