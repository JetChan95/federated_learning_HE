import tensorflow_federated as tff
import tensorflow as tf
import collections

#定义输入数据类型
BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

#定义模型类型
MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

#本地数据类型
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

#服务端模型类型
SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)

#客户端模型类型
CLIENT_MODEL_TYPE = tff.type_at_clients(MODEL_TYPE)

#客户端数据类型
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

CLIENT_INT_TYPE = tff.type_at_clients(tf.int32)

SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)