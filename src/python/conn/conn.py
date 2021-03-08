import json
import time
from collections import OrderedDict
import numpy as np
import socket
import ssl

from src.python.conn.global_ import server_global

ROOT = "L:/pycode/federated_learning_HE/src/models"
END_OF_MSG = 'msg transport complete！'
MODEL_INFO = 'MODEL_INFO_UPDATE'
REQUEST_MODEL = 'REQUEST MODEL'
MODEL_RCVED = 'MODEL_RECVED'
MODEL_ID_RCVED = 'MODEL_ID_RCVED'
TYPE_RCVED = 'TYPE_RCVED'
NO_MODEL = 'NO_MODEL'
SEND_MODEL = 'SEND_MODEL'

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

def client_query_model(id):
    """
    向服务器请求最新的model，发送本地model的id
    :param id: 本地模型的id
    :return: 成功：返回model
             失败：返回None
    """
    server_address = ('127.0.0.1',6666)
    cxt = ssl._create_unverified_context()
    #与服务器建立连接
    with socket.socket() as sock:
        with cxt.wrap_socket(sock,server_hostname = server_address[0]) as ssock:
            print("连接服务器!")
            ssock.connect(server_address)

            print("\t发送type!")
            ssock.send(REQUEST_MODEL.encode('utf-8'))
            print('\t', ssock.recv(1024).decode('utf-8'), sep='')

            print("\t发送id!")
            ssock.send(str(id).encode('utf-8'))
            print('\t', ssock.recv(1024).decode('utf-8'), sep='')

            flag = ssock.recv(1024).decode('utf-8')
            if flag == SEND_MODEL:
                print("\t检测到新模型，正在更新本地模型！")
                result = recv_model(ssock)
            elif flag == NO_MODEL:
                result = None
                print("\t已经是最新模型！")
            ssock.close()
            return result

def client_upload_model(model, id):
    """
    把本地模型发送给服务器
    :param model: 模型
    :param id:模型的id
    :return:
    """
    server_address = ('127.0.0.1',6666)
    cxt = ssl._create_unverified_context()
    #与服务器建立连接
    with socket.socket() as sock:
        with cxt.wrap_socket(sock,server_hostname = server_address[0]) as ssock:
            print("连接服务器!")
            ssock.connect(server_address)

            print("\t发送type!")
            ssock.send(MODEL_INFO.encode('utf-8'))
            print('\t', ssock.recv(1024).decode('utf-8'), sep='')

            send_model(ssock, model, id)

            ssock.close()
            return None

def send_model(conn, model, id):

    print("\t发送Model id!")
    conn.send(str(id).encode('utf-8'))
    print('\t', conn.recv(1024).decode('utf-8'), sep='')

    print("\t发送Model!")
    conn.send(encode_model(model))
    print("\t发送EDM!")
    conn.send(END_OF_MSG.encode('utf-8'))
    print('\t', conn.recv(1024).decode('utf-8'), sep='')
    # wait(1)

def server_listen_process(server_global):
    """
    监听线程，不断处理客户端连接
    :return: None
    """
    id = 0

    #创建默认上下文
    cxt = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    cxt.load_default_certs(ssl.Purpose.CLIENT_AUTH)
    #加载证书
    try:
        cxt.load_cert_chain(certfile='./py.cer',keyfile='./py.key')
    except BaseException as e:
        cxt.load_cert_chain(certfile='./src/python/server/py.cer',keyfile='./src/python/server/py.key')


    server_address = ('127.0.0.1',6666)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        #绑定本地地址
        sock.bind(server_address)
        #开始监听
        sock.listen(5)
        #将socket打包成SSL socket
        with cxt.wrap_socket(sock, server_side=True) as ssock:
            while True:
                #接收客户端连接
                client_socket, addr = ssock.accept()
                print("连接来自：",addr)
                accept_handel(client_socket, addr, server_global)
                client_socket.close()

                # #接收来自客户端的信息
                # from_client = client_socket.recv(1024)
                # print(f"receive msg from {addr}: {from_client}")
                #
                # #向客户端发信息
                # if to_client:
                #     print("***********开始向客户端发送...**************")
                #     client_socket.send(to_client.encode("utf-8"))
                #     print("***********发送结束*************************")
                # client_socket.close()




    #
    # while(True):
    #
    #     if server_global.get_recv_model_enabel() == True:
    #         print("收到模型***", id)
    #         server_global.add_model_from_client(get_model(id))
    #     id += 1
    #     time.sleep(1)
    #     if server_global.get_num_model_from_clients() == 10:
    #         print("停止接收模型")
    #         id = 0
    #         server_global.set_recv_model_enabel(False)
    #     while(server_global.get_recv_model_enabel() is False):
    #         wait(10)

def accept_handel(conn, addr, server_global):
    """
    连接处理
    :param ssock:连接的socket
    :return:
    """
    return recv_analysis(conn, addr, server_global)
    pass

def recv_analysis(conn, addr, server_global):
    """

    :param ssock:
    :return: result:解析出来的信息
    """


    try:
        flag = conn.recv(1024).decode('utf-8')
        conn.send(TYPE_RCVED.encode('utf-8'))
        print(flag)
        if flag == MODEL_INFO:
            id, model = recv_model(conn)
            print((server_global.get_recv_model_enabel() == True) \
                  and id == server_global.get_server_model_id())
            if (server_global.get_recv_model_enabel() == True)\
                    and id == server_global.get_server_model_id():
                #保存模型
                server_global.add_model_from_client(model)
                print("收到模型***", id)
            else:
                #丢弃模型
                print("无用模型，丢弃！")
            return  None

        elif flag == REQUEST_MODEL:
            return issue_model(conn, addr, server_global)
    except (ConnectionError,OSError) as e:
        print(e)
        conn.close()
        return

def issue_model(conn, addr, server_global):
    print("\t接收id:", end='')
    model_id = int(conn.recv(1024).decode('utf-8'))
    print(model_id)
    conn.send(MODEL_ID_RCVED.encode('utf-8'))
    model = OrderedDict
    server_model_id = server_global.get_server_model_id()
    if model_id < server_model_id:
        model = server_global.get_server_model()
        back = SEND_MODEL
    else:
        back = NO_MODEL
    #发送请求结果
    print("\t是否更新模型：", back == SEND_MODEL)
    conn.send(back.encode('utf-8'))

    #发送模型
    if back == SEND_MODEL:
        send_model(conn, model, server_model_id)

    return None

def recv_model(conn):
    model_info = b''
    print("\t接收id:", end='')
    model_id = int(conn.recv(1024).decode('utf-8'))
    conn.send(MODEL_ID_RCVED.encode('utf-8'))
    print(model_id)

    print("\t接收模型：", end='')
    temp = conn.recv(1024)
    while(temp.decode('utf-8') != END_OF_MSG):
        model_info += temp
        temp = conn.recv(1024)
    print("接收模型完成！")

    conn.send(MODEL_RCVED.encode('utf-8'))
    model_info = decode_model(model_info)
    # print("id:{},model{}".format(model_id, model_info))

    return model_id, model_info

def cc(msg_type, msg):

    server_address = ('127.0.0.1',6666)
    cxt = ssl._create_unverified_context()
    #与服务器建立连接
    with socket.socket() as sock:
        with cxt.wrap_socket(sock,server_hostname = server_address[0]) as ssock:
            print("连接")
            ssock.connect(server_address)
            print("发送type")
            ssock.send(msg_type.encode('utf-8'))
            print("发送msg")
            ssock.send(msg)
            print("发送EDM")
            ssock.send(END_OF_MSG.encode('utf-8'))
            wait(5)
            ssock.close()
            return None

def wait(seconds):
    time.sleep(seconds)

if __name__ == '__main__':

    sg = server_global()
    sg.update_server_model_id()
    sg.set_server_model(get_model(3))
    print(sg.get_server_model_id())
    server_listen_process(sg)