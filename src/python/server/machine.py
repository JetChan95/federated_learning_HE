import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os

from src.python.conn.conn import get_model, broadcast_model
from src.python.model.model import *
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import nest_asyncio
nest_asyncio.apply()
tff.backends.reference.set_reference_context()


from src.python.spec import *

server_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))



#聚合
def server_train(round):
    models = []
    for id in range(10):
        models.append(get_model(id))
    com_model = federated_train(models)
    broadcast_model(com_model, round)
    return com_model

print(server_train(1))