import collections
import threading

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os
import sys



sys.path.append("L:/pycode/federated_learning_HE")
from src.python.server import load_test
from src.python.conn.conn import *
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



def save_history(history, filename):
    """编码
    """
    history = history.copy()
    model = history['model']
    for key in model:
        model[key] = model[key].tolist()
    history['model'] = model
    # encoded_history = json.dumps(history, sort_keys=False, indent=4).encode('utf-8')
    encoded_model = json.dumps(model, sort_keys=False, indent=4).encode('utf-8')
    try:
        fp = open("./save/{}.loss".format(filename), 'ab')
        fp.write(history['loss'].encode('utf-8'))
        fp.close()
        fp = open("./save/{}.acc".format(filename), 'ab')
        fp.write(history['acc'].encode('utf-8'))
        fp.close()
        fp = open("./save/{}.model".format(filename), 'wb')
        fp.write(encoded_model)
        fp.close()
    except BaseException as e:
        print("保存训练过程：{}".format(e))


if __name__ == "__main__":
    print("启动模型模块……")
    sg = sg()
    print("初始化模型……")
    sg.set_server_model(server_model)
    #以守护进程启动监听
    print("启动通信模块……")
    lt = listen_thread(server_listen_process, sg)
    lt.start()
    print("加载测试数据……")
    test_data = load_test.load()
    train_historys = []
    localtime = time.localtime(time.time())
    filename = "{}{}{}{}{}".format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday,localtime.tm_hour, localtime.tm_min)
    while(True):
        if(sg.get_recv_model_enabel() == True):
            print("正在收集客户端模型***当前已就绪模型数量：{}, 模型id{}".format(sg.get_num_model_from_clients(), sg.get_server_model_id()))
            wait(2)
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
        sg.update_server_model_id()
        # print(server_model)
        #设置允许接收客户端模型
        sg.set_recv_model_enabel(True)
        #模型性能测试
        acc = cacul_acc(server_model, test_data)
        loss = local_loss(server_model, test_data)
        print("第{}轮训练，测试机准确率{}, 损失函数{}".format(sg.get_server_model_id(), acc, loss))
        history = {"model":server_model.copy(),"acc":"{}\n".format(acc),"loss":"{}\n".format(loss)}
        save_history(history, filename)






