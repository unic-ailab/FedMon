import tensorflow as tf
import numpy as np
from logging import DEBUG
import traceback
from FedMon.profiler import Profiler
from flwr.common.logger import log
import sys
import flwr as fl

import json
import os
import numpy as np
os.unsetenv("http_proxy")
os.unsetenv("https_proxy")


def dist(N=1000,nodes=10, **kwargs):
    return np.linspace(N/nodes, N/nodes, nodes)

profiler = Profiler()

def apply_distribution(x, y, dist, partition):
    prev_partition = partition - 1
    start = 0
    if prev_partition >= 0:
        start = int(dist[prev_partition])
    end = int(dist[partition])
    x = x[start: end]
    y = y[start: end]
    log(DEBUG, f"-------------------------------------")
    log(DEBUG, f"Data distribution size: {end-start}. starting point: {start}, ending point: {end} ")
    log(DEBUG, f"-------------------------------------")

    return x, y

def dataset_resizing(data,
                     partition,
                     nodes,
                     distribution_parameters={}):
    
    if partition > nodes:
        partition = nodes
    
    x, y = data
    if len(x) == 2 and len(y) == 2: # sklearn
        (x_train, y_train), (x_test, y_test) = data
        x_N = len(x_train)
        distribution_parameters['N'] = x_N
        gen_dist = dist(**distribution_parameters).cumsum()
        _x_train, _y_train = apply_distribution(x_train, y_train, gen_dist, partition)
        x_N = len(x_test)
        distribution_parameters['N'] = x_N
        gen_dist = dist(**distribution_parameters).cumsum()
        _x_test, _y_test = apply_distribution(x_test, y_test, gen_dist, partition)
        return (_x_train, _y_train), (_x_test, _y_test)

    # others
    try:
        (x_train, y_train), (x_test, y_test) = (x.data, x.targets), (y.data, y.targets)
    except Exception as ex:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        trace_back = traceback.extract_tb(ex_traceback)
        log(DEBUG, f"5.1 {dir(x)} {dir(y)} {ex_type.__name__}, {ex_value}, {trace_back}")
    x_N = len(x_train)
    distribution_parameters['N'] = x_N
    gen_dist = dist(**distribution_parameters).cumsum()
    x.data, x.targets = apply_distribution(x_train, y_train, gen_dist, partition)
    x_N = len(x_test)
    distribution_parameters['N'] = x_N
    gen_dist = dist(**distribution_parameters).cumsum()
    y.data, y.targets = apply_distribution(x_test, y_test, gen_dist, partition)

    return x, y

DATASET_DIR = "/data/tensorflow"

class TensorflowDatasetHandlerMnist(fl.client.NumPyClient):
    dataset = "mnist"
    _dataset = None

    def __init__(self, **kwargs):
        
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])

        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # fl.client.NumPyClient.__init__(self, **kwargs)

    @profiler
    def load_data(self, partition: int, nodes: int, training_set_size: int = 50_000, test_size: int = 10_000,
                  random: bool = False, distribution: str = 'flat', distribution_parameters: dict = {}):
        file = f"{DATASET_DIR}/{self.dataset}.npz"
        if self._dataset is None:
            self._dataset = tf.keras.datasets.mnist.load_data(file)

        ((x_train, y_train), (x_test, y_test)) = self._dataset
        x_train = x_train[:training_set_size]
        y_train = y_train[:training_set_size]
        x_test = x_test[:test_size]
        y_test = y_test[:test_size]
        self._dataset = ((x_train, y_train), (x_test, y_test))
        res = dataset_resizing(self._dataset,
                     partition,
                     nodes,
                     distribution=distribution,
                     distribution_parameters=distribution_parameters)
        return res

    @profiler
    def get_parameters(self, config=None):
        log(DEBUG, f"get_parameters")
        return self.model.get_weights()

    @profiler   
    def train(self, dataset, epochs, batch_size, *args, **kwargs):
        x_train, y_train = dataset
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return len(y_train)

    
    @profiler
    def test(self, dataset, *args, **kwargs):
        (x_test, y_test) = dataset
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, accuracy, len(y_test)
    
    @profiler   
    def set_model_parameters(self, parameters, *args, **kwargs):
        self.model.set_weights(parameters)

    def fit(self, parameters, config): 
        log(DEBUG, f"fit")

        if self.handler.model_handler.set_parameters_on_fit:
            self.set_parameters(parameters)

        num_of_examples = self.train()
        res = list(self.get_parameters()), num_of_examples, {}
        Profiler.store_metrics()
        return res
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_examples = self.test()
        Profiler.log_metric("accuracy", accuracy)
        Profiler.log_metric("loss", loss)
        return loss, num_examples, {"accuracy": accuracy}
    



backend = str(os.getenv("FL_BACKEND")).lower()
num_of_threads = int(os.getenv("FL_NUM_OF_THREADS", 0))
dataset = os.getenv("FL_DATASET", 'CIFAR10')
server = os.getenv("FL_SERVER", "[::]")
training_set_size = int(os.getenv("FL_TRAINING_SET_SIZE", -1))
test_set_size = int(os.getenv("FL_TEST_SET_SIZE", -1))
epochs = int(os.getenv("FL_EPOCHS", 1))
host = os.getenv("FL_HOST", "")

# new parameters for distribution
num_of_nodes = int(os.getenv("FL_NODES"))
node_id = int(os.getenv("FL_NODE_ID"))
distribution = str(os.getenv("FL_DATASET_DISTRIBUTION", 'flat'))
distribution_randomness = bool(os.getenv("FL_DATASET_RANDOM", False))  # TODO Add random as functionality
distribution_params_str = str(os.getenv("FL_DATASET_DISTRIBUTION_PARAMETERS", ''))

# distribution_params = {}
# try:
#     distribution_params = json.loads(distribution_params_str)
# except Exception:
#     pass


# dataset_params = dict(
#     partition=node_id,
#     nodes=num_of_nodes,
#     test_size=test_set_size,
#     training_set_size=training_set_size,
#     random=distribution_randomness,
#     distribution=distribution,
#     )

if __name__ == "__main__":
    dataset = dataset.lower()
    backend = backend if backend in ["pytorch", "pytorch_mobilenetv2", "tensorflow", "mxnet", "pytorch_light", "sklearn"] else "pytorch"
    backend_class = f"{backend}_{dataset}"
    fl.client.start_numpy_client(server_address=server + ":8080", client=TensorflowDatasetHandlerMnist(
                                                                    epochs=epochs,
                                                                    host=host
                                                                    ))