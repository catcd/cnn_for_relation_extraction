from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Merge
from sklearn.model_selection import KFold
from helpers import evaluate_helpers as eh
from models.model_const import BATCH_SIZE


class ModelS2345F24H288:
    def __init__(self, embed_size, sentence_length):
        """
        :param embed_size: int
        :param sentence_length: int
        """
        self.sentence_length = sentence_length
        self.embed_size = embed_size

    def cross_validation(self, x_data, y_data):
        """
        :param x_data:
        :param y_data:
        :return:
        """
        kf = KFold(n_splits=10, shuffle=True)
        ret = []
        i = 0
        for train_index, test_index in kf.split(y_data):
            i += 1

            print("10-Fold step {}".format(i))
            model = self.get_model()

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

            model.fit([x_train, x_train, x_train, x_train], y_train, batch_size=BATCH_SIZE, epochs=25)

            y_pred = model.predict_classes([x_test, x_test, x_test, x_test], batch_size=BATCH_SIZE)
            p, r, f1, _ = eh.calculate_f_measure(y_test, y_pred)
            print("p={} r={} f1={}".format(p, r, f1))
            ret.append((p, r, f1))

        return ret

    def train_separated_test(self, x_train, y_train, x_test, y_test):
        """
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        print("Training")
        model = self.get_model()

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        model.fit([x_train, x_train, x_train, x_train], y_train, batch_size=BATCH_SIZE, epochs=25, shuffle=True)

        y_pred = model.predict_classes([x_test, x_test, x_test, x_test], batch_size=BATCH_SIZE)
        p, r, f1, _ = eh.calculate_f_measure(y_test, y_pred)
        print("p={} r={} f1={}".format(p, r, f1))

        return p, r, f1

    def get_model(self):
        # Construct model
        """
        :return: Sequential
        """
        model = Sequential()

        model2 = Sequential()
        model2.add(
            Conv2D(24, (2, self.embed_size),
                   activation='relu',
                   input_shape=(self.sentence_length, self.embed_size, 1))
        )
        model2.add(MaxPooling2D(pool_size=(59, 1)))

        model3 = Sequential()
        model3.add(
            Conv2D(24, (3, self.embed_size),
                   activation='relu',
                   input_shape=(self.sentence_length, self.embed_size, 1))
        )
        model3.add(MaxPooling2D(pool_size=(58, 1)))

        model4 = Sequential()
        model4.add(
            Conv2D(24, (4, self.embed_size),
                   activation='relu',
                   input_shape=(self.sentence_length, self.embed_size, 1))
        )
        model4.add(MaxPooling2D(pool_size=(57, 1)))

        model5 = Sequential()
        model5.add(
            Conv2D(24, (5, self.embed_size),
                   activation='relu',
                   input_shape=(self.sentence_length, self.embed_size, 1))
        )
        model5.add(MaxPooling2D(pool_size=(56, 1)))

        model.add(Merge([model2, model3, model4, model5], mode='concat'))

        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model
