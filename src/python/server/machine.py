import collections
import threading

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os

from src.python.conn.conn import get_model, broadcast_model, server_listen_process, wait, global_enabel, \
    global_models_from_client
from src.python.model.model import *
from src.python.conn.global_ import server_global as sg
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
        print(len(models))
    com_model = federated_train(models)
    return com_model


class listen_thread(threading.Thread):
    def __init__(self, func, sg):
        threading.Thread.__init__(self)
        self.sg = sg
        self.func = func
        self.daemon = True
    def run(self):
        self.func(self.sg)

if __name__ == "__main__":

    sg = sg()
    sg.set_server_model(server_model)
    #以守护进程启动监听
    lt = listen_thread(server_listen_process, sg)
    lt.start()
    while(True):
        if(sg.get_recv_model_enabel() == True):
            print("没有足够的模型用以聚合，请等待***")
            wait(5)
            continue
        #取出客户端模型
        # print("取出模型***")
        models = sg.get_models_from_clients()
        print(len(models))
        #清空全局变量
        sg.clear_models_from_clients()
        #模型聚合
        print("开始聚合***")
        server_model = federated_train(models).copy()
        #更新模型
        print("更新模型***")
        sg.set_server_model(server_model)
        print(server_model)
        #设置允许接收客户端模型
        sg.set_recv_model_enabel(True)




