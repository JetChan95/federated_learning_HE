import collections
import threading
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os
import sys
sys.path.append("L:/pycode/federated_learning_HE")

from src.python.client import load_data
from src.python.conn.conn import client_get_model, client_query_model, client_upload_model, wait, upload_model

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

# #加载数据
# train, test = load_data.load(NUM_SAMPLES)
# print(train[1])
#
# #加载训练好的模型
# trained_model = client_get_model(999)
#
# print("",trained_model)
#
# ##########################################################
# #初始化模型
initial_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))
#
# sample_batch = train[1]
# ##########################################################
# loss = batch_loss(initial_model.copy(), sample_batch)
# print("测试损失函数",loss)
# ##########################################################
# model = batch_train(initial_model.copy(), sample_batch, 0.1)
# print("测试batch训练",model)
# ##########################################################

# models = list()
# model = initial_model
# for _ in range(10):
#     train, test = load_data.load(NUM_SAMPLES)
#     model = local_train(initial_model.copy(), 0.1, train)
#     upload_model(model, _)
#     print("测试local训练",model)
#     models.append(model)
# com_  = federated_train(models)
# print(com_)
# ##########################################################
# acc = cacul_acc(trained_model, test)
# print("测试准确率:{}".format(acc))

#学习率
LR = 0.1
#客户端模型
client_model = collections.OrderedDict
client_model_id = -1

class client():
    def __init__(self, cid, lr=0.1, client_model=collections.OrderedDict,  client_model_id=-1):

        self.lr = lr
        self.client_model = client_model
        self.client_model_id = client_model_id
        self.cid = cid


    def run(self):
        while(True):
            query_result = client_query_model(self.client_model_id)
            if (query_result is None):
                wait(10)
                print("没有可用模型，请等待！")
                continue
            #更新本地模型
            self.client_model_id, self.client_model = query_result
            print(self.client_model)
            #加载数据
            train, test = load_data.load(NUM_SAMPLES)
            #训练模型
            self.client_model = local_train(self.client_model, self.lr, train)
            print(self.client_model)
            #上传模型
            client_upload_model(self.client_model, self.client_model_id)


if __name__ == "__main__":

    c = client(1)
    c.run()
    # while(True):
    #     query_result = client_query_model(-1)
    #     if (query_result is None):
    #         wait(10)
    #         print("没有可用模型，请等待！")
    #         continue
    #     #更新本地模型
    #     client_model_id, client_model = query_result
    #     print(client_model)
    #     #加载数据
    #     train, test = load_data.load(NUM_SAMPLES)
    #     #训练模型
    #     client_model = local_train(client_model, LR, train)
    #     print(client_model)
    #     #上传模型
    #     client_upload_model(client_model, client_model_id)
    #     # wait(10)

