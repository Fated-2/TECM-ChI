import keras
from keras import Input, Model, optimizers
from keras.layers import Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Dropout, Layer, Embedding, Concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
from keras.layers import GlobalMaxPooling1D
from keras import backend as K
from keras import initializers
from keras.layers import LSTM, BatchNormalization, Bidirectional, Add


class AttLayer(Layer):
    """
    自定义的注意力机制层, 继承了Keras的Layer类
    通过学习输入序列中每个元素的重要性来加权输入, 然后对加权后的输入进行求和,得到最终输出
    """

    # 初始化(构造函数), 接受了一个参数attention_dim指定注意力层的维度(用于表示注意力权重的向量的维度大小)
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True  # 支持掩码, 掩码可用来指示序列中哪些位置是有效的
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()  # 调用执行父类的初始化方法

    # 初始化层的权重, 这些权重在模型训练过程中会被学习和更新
    def build(self, input_shape):   # shape(,830,128)
        assert len(input_shape) == 3  # 输入形状的长度是3, 代表批次大小、时间步长和特征数量
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))  # 创建一个权重矩阵W(128, 50)
        self.b = K.variable(self.init((self.attention_dim,)))  # 创建一个偏置向量b(50,), 形状即注意力维度
        self.u = K.variable(self.init((self.attention_dim, 1)))  # 创建一个权重矩阵, 形状(注意力维度, 1), 用于计算注意力分数
        self.trainable_weights = [self.W, self.b, self.u]  # 这个列表会在训练过程中被优化器用来更新权重
        super(AttLayer, self).build(input_shape)  # 调用父类的build, 正确完成层的构建过程

    # 当输入数据具有掩码时, 才会被调用, mask是可选参数, 表示输入数据的掩码
    def compute_mask(self, inputs, mask=None):
        return mask

    # 通过计算输入数据x的加权和来生成输出, x的形状为 [batch_size, sel_len, attention_dim]，
    # 其中 batch_size 是批次大小，sel_len 是序列长度，attention_dim 是注意力维度
    def call(self, x, mask=None):   # shape(,830,128)
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        # 先计算输入x与权重矩阵W的点积, 再加上偏置b, 再用Tanh 激活函数来获得中间表示uit
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))  # shape(,830,50)
        # 计算注意力分数, ait代表注意力权重
        ait = K.dot(uit, self.u)  # shape(,830,1)将中间表示与权重向量点积, 得到注意力分数的原始值
        ait = K.squeeze(ait, -1)    #shape(,830)
        # 应用掩码
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        # 归一化注意力权重
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # 加权输入
        ait = K.expand_dims(ait)    # shape(,830,1)
        weighted_input = x * ait    # shape(,830,128)
        # 计算输出, 通过这一函数模型就可以学习输入序列中不同部分的重要性, 并在生成输出时给予不同的权重
        output = K.sum(weighted_input, axis=1)  # shape(,128)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def CMANet(input_shape_sequence, input_shape_genomics) :

    sequence_input = keras.layers.Input(shape=input_shape_sequence)
    input_data_genomics = keras.layers.Input(shape=input_shape_genomics)

    output = keras.layers.Conv1D(101, kernel_size=9, strides=1, activation='relu')(
        sequence_input)  # shape(,5000-9+1=4992,32)
    output = keras.layers.AveragePooling1D(pool_size=3, strides=3)(output)  # shape(,4992/3=1664,32)
    output = keras.layers.Conv1D(128, kernel_size=5, strides=1, activation='relu')(output)  # shape(,1664-5+1=1660,64)
    output = keras.layers.AveragePooling1D(pool_size=2, strides=2)(output)  # shape(,1660/3=553,64)
    output = keras.layers.Dropout(0.5)(output)  # shape(,553,64)

    # 添加双向GRU层Bidirectional(GRU)到模型中，隐藏层大小为hiddensize，设置返回完整序列。
    # output = keras.layers.Bidirectional((keras.layers.GRU(120, return_sequences=True)))(output)   # shape(,553,240)
    # output = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(output)
    # output = keras.layers.Dropout(0.5)(output)
    # # 添加展平层Flatten到模型中，用于将输入展平为一维向量。
    # output = keras.layers.Flatten()(output)   # shape(,553*240=132720)

    # 注意力机制
    output = AttLayer(50)(output)  # 输入数据的shape(batch, 831, 128)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.Dense(64, activation='relu')(output)  # shape(,32)
    # output = keras.layers.Dropout(0.25)(output)   # shape(,132720)

    # 对输入的基因组特征数据进行批量归一化, 加速训练过程, 提高泛化能力
    merge2 = keras.layers.BatchNormalization()(input_data_genomics)
    merge2 = keras.layers.Dropout(0.5)(merge2)
    merge2 = keras.layers.Dense(128, activation='relu')(merge2)
    # 合并正反向DNA序列和基因组特征
    merge2 = keras.layers.Concatenate(axis=1)([output, merge2])

    output = keras.layers.Dense(2, activation='softmax')(merge2)

    model = keras.models.Model(inputs=[sequence_input, input_data_genomics], outputs=output)
    # print('打印模型主要信息 : ', model.summary())  # 打印模型的主要信息, 层的数量, 参数数量
    return model

def IChrom_genomics(input_shape_genomics):
    input_data_genomics = Input(shape=input_shape_genomics)

    merge2 = BatchNormalization()(input_data_genomics)
    merge2 = Dropout(0.5)(merge2)
    merge2 = Dense(128, activation='relu')(merge2)

    # output = Dense(16, activation='sigmoid')(merge2)
    output = Dense(1, activation='sigmoid')(merge2)

    model = Model(input_data_genomics, output)
    print(model.summary())
    return model
