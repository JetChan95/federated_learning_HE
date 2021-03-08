import tensorflow_federated as tff
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import nest_asyncio
nest_asyncio.apply()
tff.backends.reference.set_reference_context()

from src.python.spec import *


#一个batch的学习损失函数
@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias'])
    return -tf.reduce_mean(
        tf.reduce_sum(
            tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)

#一个batch的梯度下降算法
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`. Must
    # be defined outside the @tf.function.
    model_vars = collections.OrderedDict([
        (name, tf.Variable(name=name, initial_value=value))
        for name, value in initial_model.items()
    ])
    #优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        # 用`batch_loss`计算一次损失作为单次梯度
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
        return model_vars

    return _train_on_batch(model_vars, batch)


#本地训练算法
@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)

#模型度量参数loss
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_loss(model, all_batches):
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))

#模型度量参数acc
@tf.function
def pre_(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias'])
    pre = tf.cast(tf.argmax(predicted_y, axis=1), dtype=tf.int32)
    real = batch['y']
    correct = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(pre, real), dtype=tf.float32)))
    correct = tf.reshape(correct, [1])
    total = tf.cast(tf.shape(pre), dtype=tf.float32)
    acc = tf.math.divide(correct, total)
    acc_ = tf.concat([correct, total], axis=0)
    return acc_
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_pre(model, batch):
    return pre_(model, batch)
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_acc(model, all_batches):
    acc = tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_pre(model, b), BATCH_TYPE),
            all_batches))

    acc_l = tff.sequence_map(
        tff.federated_computation(lambda b: batch_pre(model, b), BATCH_TYPE),
        all_batches)
    return acc


#计算准确率
def cacul_acc(model, data):
    acc = local_acc(model, data)
    out = tf.dynamic_partition(acc, [0, 1], 2)
    acc = tf.reduce_sum(tf.divide(out[0], out[1]))
    return acc

#加载客户端上传的模型
@tff.tf_computation(MODEL_TYPE)
def load_model(model):
    return model

#模型聚合
@tff.federated_computation(CLIENT_MODEL_TYPE)
def federated_train(models):
    return tff.federated_mean(
        tff.federated_map(load_model,models))
