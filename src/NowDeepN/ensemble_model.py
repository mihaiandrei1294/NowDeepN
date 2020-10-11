import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
from imblearn.over_sampling import SMOTE
from keras.models import Sequential, model_from_json
import numpy as np
import time
from multiprocessing import Process, Lock
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

from NowDeepN.network_models import get_ensemble_network_model


class EnsembleNetwork:

    def __init__(self, input_dim: int, number_of_networks: int, name: str, new_model = True, output_dim = 1):
        self.__input_dim = input_dim
        self.__number_of_networks = number_of_networks
        self.__name = name
        self.__models = []
        self.__mutex_original = Lock()
        self.__mutex_current = Lock()
        self.__new = new_model
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models/" + name):
            os.makedirs("models/" + name)
        self._loss = "mean_squared_error"
        self.__output_dim = output_dim

    def get_input_dim(self):
        return self.__input_dim

    def get_number_of_networks(self):
        return self.__number_of_networks

    def get_output_dim(self):
        return self.__output_dim

    def __get_weak_model(self):
        input_dim = self.__input_dim

        model = get_ensemble_network_model(self.__name, input_dim, self.__output_dim)
        self.__compile_model(model)
        return model

    def __compile_model(self, model: Sequential):
        self._loss = "mean_squared_error"
        model.compile(loss=self._loss, optimizer="adam")


    def work(self, i, train_data:np.ndarray, train_result:np.ndarray, epochs = 30, batch_size = 1000, smote = False):
        # print("HELLO BEFORE SESSION SET")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU


        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        # print("HELLO AFTER SESSION SET")

        current_train_data = train_data
        current_train_result = train_result
        # print("train_data.shape=",current_train_data.shape)
        # print("train_result.shape=",current_train_result.shape)

        if self.__new:
            # print("IS NEW")
            model = self.__get_weak_model()
        else:
            # print("IS_OLD")
            # load json and create model
            json_file = open('models/' + self.__name + '/' + str(i) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("models/" + self.__name + "/" + str(i) + ".h5")
            self.__compile_model(model)


        # print("Model ",i + 1," starting to train!")

        if smote:
            start = time.time()
            with self.__mutex_original:
                null_values_labels = (train_data[:, i] != 0).astype(int)
                sm = SMOTE(n_jobs=32)
                x = np.concatenate((train_data, train_result), axis=1)
                print("no. of smaples before smote: ", x.shape[0], "no. of non-zero values before smote: ",
                      null_values_labels.sum(), flush=True)
                if null_values_labels.sum() != 0:
                    x, y = sm.fit_resample(x, null_values_labels)
                    print("no. of smaples after smote: ", x.shape[0], "no. of non zero values after smote: ", y.sum(), flush=True)
                with self.__mutex_current:
                    current_train_data, current_train_result = x[:, :self.__input_dim], x[:, self.__input_dim:]
                    assert (current_train_data.shape[1] == self.__input_dim)
                    assert (current_train_result.shape[1] == self.__number_of_networks)
            end = time.time()
            print("SMOTE duration: ", end - start, flush=True)

        y = current_train_result[:, i]
        # print("y.shape=",y.shape)

        # model = self.__model[i]
        # print("		training model ", i+1, " of 13", file=sys.stderr, flush = True)
        start = time.time()
        model.fit(current_train_data, y, epochs=epochs, batch_size=batch_size, verbose=0)
        # model.fit(current_train_data, y, epochs=epochs, batch_size=batch_size, verbose=1)
        end = time.time()
        print("     trained model ", i + 1, " of 13 in ", end - start, " seconds", flush = True)

        # SAVE MODEL
        model_json = model.to_json()
        with open("models/" + self.__name + "/" + str(i) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        ais = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        model.save_weights("models/" + self.__name + "/" + ais[i] + ".h5")
        # print("Saved model to disk")

        # Free memory
        del model
        clear_session()
        gc.collect()

    def __parallel_fit(self, train_data, train_result, epochs = 30, batch_size = 1000, smote = False, to_fit = None):
        start = time.time()
        if to_fit is None:
            to_fit = range(0, self.__number_of_networks)

        self.__models.clear()
        processes = []
        # print("Train data is:  ", train_data)
        # print("Type of train data is:  ", type(train_data))
        if type(train_data) == type([]):
            for i in to_fit:
                train_data_i = train_data[i]
                p = Process(target=EnsembleNetwork.work,
                            args=(self, i, train_data_i, train_result[i], epochs, batch_size, smote))
                p.start()
                processes.append(p)
        else:
            for i in to_fit:
                train_data_i = train_data
                p = Process(target=EnsembleNetwork.work, args=(self, i, train_data_i, train_result, epochs, batch_size, smote))
                p.start()
                processes.append(p)
        # print("Started All processes")
        for p in processes:
            p.join()

        end = time.time()
        print("time to train on timestamp: ", end - start, " seconds", flush=True)
        self.__new = False




    def __non_parallel_fit(self, train_data:np.ndarray, train_result:np.ndarray, epochs = 30, batch_size = 1000, smote = False, to_fit = None):
        # train data should be of shape (no_of_instances, input_dim)
        # train result should be of shape (no_of_instances, output_dim)
        # print("\n Fit epoch ", e+1,"/", epochs)

        if to_fit is None:
            to_fit = range(0, self.__number_of_networks)
        total_time = 0

        for i in to_fit:
            y = train_result
            x = train_data
            if type(train_data) == type([]):
                y = train_result[i]
                x = train_data[i]
            # print("     training model ", i+1, " of ", self.__output_dim)
            start = time.time()
            self.work(i, x, y, epochs, batch_size, smote)
            end = time.time()
            total_time += end - start
            print("     trained model ", i + 1, " of ", self.__number_of_networks, " in ", end - start, " seconds", flush=True)
        print("time to train epoch: ", total_time, " seconds", flush=True)

        # Free memory
        self.__new = False
        del self.__models[:]
        gc.collect()

    def fit(self, train_data:np.ndarray, train_result:np.ndarray, epochs = 30, batch_size = 1000, smote = False, parallel = True, to_fit = None):
        if parallel:
            self.__parallel_fit(train_data, train_result, epochs, batch_size, smote, to_fit)
        else:
            self.__non_parallel_fit(train_data, train_result, epochs, batch_size, smote, to_fit)


    def predict(self, data):
        if len(self.__models) < 1:
            self.load_model()
        if type(data) == type([np.array([1])]):
            results = []
            for i in range(0, self.__number_of_networks):
                model = self.__models[i]
                input_data = data[i]
                print("Predicting for model ", i + 1, flush=True)
                pred = model.predict(input_data)
                if pred == []:
                    pred = np.empty((input_data.shape[0],0))
                if self.__output_dim > 1:
                    pred = pred.reshape((pred.shape[0], self.__output_dim))
                else:
                    pred = pred.reshape((pred.shape[0],))
                results.append(pred)
        else:
            results = np.zeros((data.shape[0], self.__number_of_networks))
            if self.__output_dim > 1:
                results = np.zeros((data.shape[0], self.__number_of_networks, self.__output_dim))
            for i in range(0, self.__number_of_networks):
                model = self.__models[i]
                input_data = data
                print("Predicting for model ", i + 1, flush=True)
                pred = model.predict(input_data)
                if self.__output_dim > 1:
                    pred = pred.reshape((pred.shape[0], self.__output_dim))
                else:
                    pred = pred.reshape((pred.shape[0],))
                results[:, i] = pred
        return results

    def load_model(self, path = 'models/'):
        self.__models = []
        for i in range(0, self.__number_of_networks):
            # load json and create model
            json_file = open(path + self.__name + '/' + str(i) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(path + self.__name + "/" + str(i) + ".h5")
            self.__models.append(model)
        print("Loaded model from disk", flush=True)

    def save_model(self):
        for i in range(0, self.__number_of_networks):
            model = self.__models[i]
            model_json = model.to_json()
            with open("models/" + self.__name + "/" + str(i) + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("models/" + self.__name + "/" + str(i) + ".h5")
        # print("Saved model to disk")