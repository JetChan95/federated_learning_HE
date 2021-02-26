import json
import time
from collections import OrderedDict
import numpy as np

ROOT = "L:/pycode/federated_learning_HE/src/models"

def encode_model(model, encoding = 'utf-8'):
    """编码
    """
    model_vars = model.copy()
    for key in model_vars:
        model_vars[key] = model_vars[key].tolist()
    encoded_model = json.dumps(model_vars, sort_keys=False, indent=4).encode(encoding)
    return encoded_model

def decode_model(encoded_model, encodng='utf-8'):
    """解码
    """
    model_vars = json.loads(encoded_model, object_pairs_hook=OrderedDict)
    for key in model_vars:
        model_vars[key] = np.array(model_vars[key], dtype='float32')
    return model_vars

def upload_model(model, id):
    """上传本地model参数到服务器
    目前使用文件系统模拟"""
    model = model
    client_model_path = ROOT + "/client/{}.modle".format(id)

    with open(client_model_path, 'wb') as fp:
        fp.write(encode_model(model))
        fp.close()

def get_model(id):
    """获取客户端上传的model参数"""
    client_model_path = ROOT + "/client/{}.modle".format(id)
    with open(client_model_path, 'rb') as fp:
        lines = fp.read()
        fp.close()
    return(decode_model(lines))

def broadcast_model(model, round):
    model = model.copy()
    model = encode_model(model)
    with open(ROOT+"/broadcast/{}.model".format(round), 'wb') as fp:
        fp.write(model)
        fp.close()

def client_get_model(round):
    with open(ROOT+"/broadcast/{}.model".format(round), 'rb') as fp:
        model = fp.read()
        fp.close()
    return decode_model(model)

def client_query_model():
    """
    向服务器请求最新的model，发送本地model的id
    :param id: 本地模型的id
    :return: 成功：返回model
             失败：返回None
    """

    return None

def client_upload_model(model, id):
    """
    把本地模型发送给服务器
    :param model: 模型
    :param id:模型的id
    :return:
    """

global_enabel = True
global_num_model_from_client = 0
global_models_from_client = list()
global_server_model = None

def server_listen_process(server_global):
    """
    监听线程，不断处理客户端连接
    :return: None
    """
    id = 0
    while(True):

        if server_global.get_recv_model_enabel() == True:
            print("收到模型***", id)
            server_global.add_model_from_client(get_model(id))
        id += 1
        time.sleep(1)
        if server_global.get_num_model_from_clients() == 10:
            print("停止接收模型")
            id = 0
            server_global.set_recv_model_enabel(False)
        while(server_global.get_recv_model_enabel() is False):
            wait(10)

def wait(seconds):
    time.sleep(seconds)