import json
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

def upload_model(model, dest):
    """上传本地model参数到服务器
    目前使用文件系统模拟"""
    model = model
    client_model_path = dest

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