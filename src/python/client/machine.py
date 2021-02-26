import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os

from src.python.client import load_data
from src.python.conn.conn import client_get_model

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import nest_asyncio
nest_asyncio.apply()
tff.backends.reference.set_reference_context()

from src.python.model.model import *


#数据量
NUM_SAMPLES = 1000



def cacul_acc(model, data):
    acc = local_acc(model, data)
    out = tf.dynamic_partition(acc, [0, 1], 2)
    acc = tf.reduce_sum(tf.divide(out[0], out[1]))
    return acc

#加载数据
train, test = load_data.load(NUM_SAMPLES)
print(train[1])

#加载训练好的模型
trained_model = client_get_model(999)

print("",trained_model)

##########################################################
#初始化模型
initial_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))

sample_batch = train[1]

loss = batch_loss(initial_model.copy(), sample_batch)
print("测试损失函数",loss)
##########################################################
model = batch_train(initial_model.copy(), sample_batch, 0.1)
print("测试batch训练",model)
##########################################################
model = local_train(initial_model, 0.1, train)
print("测试local训练",model)
##########################################################
acc = cacul_acc(trained_model, test)
print("测试准确率:{}".format(acc))