import tensorflow as tf
import time
import os
import collections
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.disable_eager_execution()
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)#不使用科学计数法表示浮点数
#参数设置
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
# dropout_rate = 0.1
seq_length=35
embedding_size = 512
batch_size = 128
#方法转换为索引，词向量
def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = collections.Counter(sentences)  # [('a', 5), ('r', 2), ('b', 2)]
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]  # 次数从大到小列出方法名x[0]为方法名,即[‘a’,'r','b']
  vocabulary_inv = list(sorted(vocabulary_inv))  # 列出字母排序后的方法列表[a','b','r']
  # Mapping from word to index
  vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}  # i是开始序号，x为方法名,结果字典{'a':0,'r':1,'b':2}
  return [vocabulary, vocabulary_inv]


with open('data/ZipOutputStream/input_traces/input.txt', "r") as f:  # 只读打开input.txt
  data = f.read()
with open('data/ZipOutputStream/input_traces/cluster_traces.txt', "r") as s:  # 只读打开input.txt
  test = s.read()
  # Optional text cleaning or make them lower case, etc.
  # data = self.clean_str(data)
x_text = data.split()  # 以空格为分隔符，轨迹的方法列表
#print(x_text)
test = test.split()
# print(data)
vocab, words = build_vocab(x_text)  # 字-索引映射{‘a’:0,'r':1,'b':2}，索引-字映射['a','b','r']
#print('x_text',len(x_text))
#print(x_text)#['<START>', 'LinkedList',
print('vocab', vocab)  # 元组vocab {'<END>': 1, '<START>': 2, 'atwo': 3, 'bthree': 4, 'cfour': 5, 'dfive': 6} 序列用的vocab，不用担心
print('words', words)#words ['<END>', '<START>', 'atwo', 'bthree', 'cfour', 'dfive']
# print(self.vocab.get('<END>'))
# print(self.words)方法列表['<END>', '<START>', 'StackAr', 'makeEmpty', 'push', 'top', 'topAndPop']
vocab_size = len(words)  # 字-索引词汇表大小=索引长度=方法个数7

tensor = np.array(list(map(vocab.get, x_text)))  # 对x_text轨迹的方法列表进行vocab.get操作并转换为列表，np.array将数据转换为数组
test = np.array(list(map(vocab.get, test)))
#print(tensor)  # 方法对应的索引序列[1 2 5 3 6 5 3 0 1 2 3 4 6 5 4 5 0 1 2 3 4 6 5 0]
# print(self.tensor.shape)
tensor_size = tensor.size
# np.array([1, 4, 9, 16, 25])列表得到数组array([ 1,  4,  9, 16, 25])
result = list()
sub_list = list()
for item in tensor:
  if item == vocab.get('<END>'):

    sub_list.append(item)
    result.append(sub_list)
    # result.append(/n)
    sub_list = list()
  else:
    sub_list.append(item)
#print('xdata',result)#方法对应的索引序列[2, 3, 4, 5, 5, 4, 5, 6, 5, 4, 1]
#tensor = result

ydata = []
for item in result:
  sub_list = item[1:]
  ydata.append(sub_list)



#print('test',test)
result2 = []
sub_list1 = list()
for item in test:
  # print('item',item)
  # print('<END>',vocab.get('<END>'))
  if item == vocab.get('<END>'):
    #print('item',item)
    sub_list1.append(item)
    result2.append(sub_list1)
    #print('result2',result2)
    # result.append(/n)
    sub_list1 = list()
  else:
    sub_list1.append(item)
test1=result2
xdata=[result[i][0:seq_length-2] for i in range(len(result))]#截断长序列
ydata=[ydata[i][0:seq_length-2] for i in range(len(ydata))]#截断长序列
test1=[test1[i][0:seq_length-2] for i in range(len(test1))]

y=[]
for item in ydata:
  #print('item1',item)
  item=[len(words)+3]+item+[len(words)+4]
  y.append(item)
ydata=y
#print('ydata',ydata) len(words)=6  tar [9,...,10]
#input加入开始、结束符号
x=[]
for item in xdata:
  #print('item1',item)
  item=[len(words)+1]+item+[len(words)+2]
  x.append(item)
  #print('item2', item)
tensor = x#len(words)=6 [7,...,8]]
np.save('tensor_file', tensor)
xdata = tensor

test2=[]
for item in test1:
  item = [len(words) + 1] + item + [len(words) + 2]
  test2.append(item)
#print('xdata',tensor)
#测试数据处理，分段####################
# print('test',test)
# result2 = []
# sub_list1 = list()
# for item in test:
#   # print('item',item)
#   # print('<END>',vocab.get('<END>'))
#   if item == vocab.get('<END>'):
#     #print('item',item)
#     sub_list1.append(item)
#     result2.append(sub_list1)
#     #print('result2',result2)
#     # result.append(/n)
#     sub_list1 = list()
#   else:
#     sub_list1.append(item)
# test1=result2
#print('test1',test1)[[2, 3, 1], [2, 3, 4, 4, 4, 1], [2, 3, 4, 4, 6, 1]
# test = []
# for item in test1:
#     sub_list = item[1:] + [0]
#     test.append(sub_list)
# print('test',test)
# Save the data to data.npy
np.save('tensor_file', tensor)
xdata = tensor  # 方法序列数组xdata和ydata一样
# xdata=[xdata[i][0:seq_length] for i in range(len(xdata))]#截断长序列
# ydata=[ydata[i][0:seq_length] for i in range(len(ydata))]#截断长序列
# test1=[test1[i][0:seq_length] for i in range(len(test1))]
num_batches = int(len(xdata) // batch_size)  # //256
print('num_batches', num_batches)  # 15
# if num_batches == 0:
#     assert False, "Not enough data. Make seq_length and batch_size small."
# xdata=[xdata[i][0:seq_length] for i in range(len(xdata))]#截断长序列
# ydata = []
# for item in xdata:
#   sub_list = item[1:] + [0]
#   ydata.append(sub_list)

#print('ydata',ydata)
item_tail = [0]
us_len1 = [len(upois) for upois in xdata]
us_len2 = [len(upois) for upois in ydata]
us_test = [len(upois) for upois in test2]
# xdata=[[1]*le+[00]*(self.seq_length-le)for le in xdata]
xdata = [upois + item_tail * (seq_length - le) for upois, le in zip(xdata, us_len1)]
ydata = [upois + item_tail * (seq_length - le) for upois, le in zip(ydata, us_len2)]
test = [upois + item_tail * (seq_length - le) for upois, le in zip(test2, us_test)]
#print('xdata',xdata)
test = np.array(test)
xdata = np.array(xdata)
ydata = np.array(ydata)
#print('test',test)[[7 2 3 1 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
s = []
for i in range(0, int(len(xdata)) + 1, batch_size):#步长batch_size
  c = xdata[i:i + batch_size]
  s.append(c)

x_batches = s  # np.array_split(xdata, num_batches, 0)

v = []
for i in range(0, int(len(ydata)) + 1, batch_size):
  d = ydata[i:i + batch_size]
  v.append(d)
y_batches = v
#################test_batches#######################################
# u = []
# for i in range(0, int(len(test)) + 1, batch_size):
#   d = test[i:i + batch_size]
#   u.append(d)
# test_batches = u
#print('x_batches', x_batches)
#print('test_batches', test_batches)
#位置编码，提供单词在句子中的相对位置信息#################################################
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

#Masking屏蔽序列中为pad的令牌，mask为0，只输出为1的位置########################################
def create_padding_mask(seq):#将序列中为0的数为1，其余为0
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)#tf.cast数据类型转换：seq=0为true，转换为float32数据格式为1
  # add extra dimensions to add the padding添加额外的维度为pad
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
# create_padding_mask(x)#[[[[0. 0. 1. 1. 0.]]] [[[0. 0. 0. 1. 1.]]] [[[1. 1. 1. 0. 0.]]]]shape=(3, 1, 1, 5), dtype=float32
# print(create_padding_mask(x))
#look-ahead掩码可以屏蔽未来tokens
def create_look_ahead_mask(size):#3下三角
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)#保留非主对角线的元素，下方对角线元素全部保留，上方只保留0行（不保留）
  return mask  # (seq_len, seq_len)

#缩放点积attention，三个输入Q查询K键V值，按维度的平方根进行缩放，因为较大的维度，点积大小增大，导致softmax函数往很小的梯度的方向靠拢，使softmax非常硬
#例如，假设Q和K的平均值为0，方差为1。它们的矩阵乘法的平均值为0，方差为dk。所以dk的平方根用于缩放，所以不管dk的值是多少，都可以得到一致的方差。
# 如果方差太低，则输出可能太平缓，无法有效优化。如果方差太大，softmax可能会在初始化时饱和，从而难以学习。
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.计算注意力权重
  q, k, v must have matching leading dimensions.q、k、v有匹配的前置维度
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.k、v倒数第二个维度匹配
  The mask has different shapes depending on its type(padding or look ahead) mask根据pad或者look_ahead有不同的形状
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # transpose_b=True给k转置 (..., seq_len_q, seq_len_k)

  # scale matmul_qk缩放点积attention
  dk = tf.cast(tf.shape(k)[-1], tf.float32)#转换为float32数据格式
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  #编码器scaled (32, 8, 25, 25)  mask (32, 1, 1, 25)
  #解码器scaled (1, 8, 800, 800) mask (1, 1, 1, 32, 32)
  # print('scaled',scaled_attention_logits.shape)#[1,8,800,800]
  # print('mask',mask.shape)
  # add the mask to the scaled tensor.给缩放点积加入掩码mask
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)
  #800
  # softmax is normalized on the last axis (seq_len_k) so that the scores在最后一个轴上seq_len_k标准化,以便于softmax加起来等于1
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  #各个权重占比，重要程度，相加=1
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  #加权后
  return output, attention_weights
# #缩放点积attention的打印输出
# def print_out(q, k, v):
#   temp_out, temp_attn = scaled_dot_product_attention(
#       q, k, v, None)
#   print('Attention weights are:')
#   print(temp_attn)
#   print('Output is:')
#   print(temp_out)

#每个多头attention块有三个输入，Q、K、V，通过线性层分割他们并输入到不同层，降维，总的计算代价与全维的单个头部注意力相同。
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads#头
    self.d_model = d_model#模型维度

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads#划分后的维度
    #tf.keras.layers.Dense相当于在全连接层中添加一个层
    self.wq = tf.keras.layers.Dense(d_model)#输出维度大小为d_model，改变inputs的最后一维
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    #tf.reset_default_graph()
    # self.wq = tf.compat.v1.get_variable("wq", [d_model])  # [32,方法数]
    # self.wk = tf.compat.v1.get_variable("wk", [d_model])  # [32,方法数]
    # self.wv = tf.compat.v1.get_variable("wv", [d_model])  # [32,方法数]
    #softmax_b = tf.compat.v1.get_variable("softmax_b", [args.vocab_size])  # 方法数
    self.dense = tf.keras.layers.Dense(d_model)
    #tf.compat.v1.reset_default_graph()
    #self.dense = tf.compat.v1.get_variable("dense", [d_model])

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    #print('split前')
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    #print('split后')
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):##output,第一块combined_mask
    batch_size = tf.shape(q)[0]
    #print('多头，qkv')
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    #print('wq后')
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    #print('多头，qkv，划分前')
    # q = tf.Variable(self.split_heads(q, batch_size))  # (batch_size, num_heads, seq_len_q, depth)
    # k = tf.Variable(self.split_heads(k, batch_size))  # (batch_size, num_heads, seq_len_k, depth)
    # v = tf.Variable(self.split_heads(v, batch_size))  # (batch_size, num_heads, seq_len_v, depth)
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)
    #print('多头，qkv，划分后')
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

#编码器解码器部分####################################################################
#输入语句通过N个编码器层，该编码器层为序列中的每个字/令牌生成输出。
#解码器关注编码器的输出和自己的输入（自我关注）来预测下一个单词。
#编码器，两层：多头注意、前向网络，层间有残差连接和层归一化，残差连接有助于避免梯度消失
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    #前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
  def call(self, x, training, mask):
    #print('encode——layer，attention前')
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    #print('encode——layer，attention后')
    attn_output = self.dropout1(attn_output, training=training)
    #print('encode——layer，dropout后')
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    #print('encode——layer，layernorm后')
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    #print('encode——layer，ffn后')
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2
# sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
# sample_encoder_layer_output = sample_encoder_layer(
#     tf.random.uniform((64, 43, 512)), False, None)
#
# sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)
#解码器
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):#output,enc_output,False,第一块combined_mask,第二块dec_padding_mask
    # enc_output.shape == (batch_size, input_seq_len, d_model)
#output,第一块combined_mask
    # print('x=output',x.shape)#x=output (1, 32, 25, 128)  (32, 1, 128)
    # print('enc_output',enc_output.shape)#enc_output (32, 25, 128)
    # print('第一块mask',look_ahead_mask.shape)#第一块mask(1, 1, 1, 32, 32) (32, 1, 1, 1)
    # print('第二块mask',padding_mask.shape)#(32, 1, 1, 25)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2
#编码器，输入嵌入、位置编码、N个编码器层
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)#输入词汇大小
    self.pos_encoding = positional_encoding(maximum_position_encoding,#最大位置编码？？？
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  # encoder_input(1,25),False,enc_padding_mask(1, 1, 1, 25)
  def call(self, x, training, mask):
    with tf.compat.v1.variable_scope("encoder"):
      seq_len = tf.shape(x)[1]
      #print('encode，embedding前',seq_len)#25
      # adding embedding and position encoding.
      #print('input_vocab_size',input_vocab_size)#9
      #print('d_model',d_model)#512
      #print('x',x.shape)
      #print(x)
      x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
      #print('encode，embedding后')
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      #print('ppp', x.shape)#32,25,128
      x += self.pos_encoding[:, :seq_len, :]
      #print('encode，pos——embedding后')
      #print('pppp',x.shape)
      x = self.dropout(x, training=training)
      #print('encode，dropout后')
      for i in range(self.num_layers):
        with tf.compat.v1.variable_scope("num_blocks_{}".format(i)):
          x = self.enc_layers[i](x, training, mask)
      #print('encode，层循环后')
    return x  # (batch_size, input_seq_len, d_model)

#解码器
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):#output,enc_output,False,第一块combined_mask,第二块dec_padding_mask
    with tf.compat.v1.variable_scope("decoder"):
      seq_len = tf.shape(x)[1]
      attention_weights = {}

      x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x += self.pos_encoding[:, :seq_len, :]

      x = self.dropout(x, training=training)
      #
      for i in range(self.num_layers):
        with tf.compat.v1.variable_scope("num_blocks_{}".format(i)):
          x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)#第一块combined_mask,第二块dec_padding_mask

          attention_weights[f'decoder_layer{i+1}_block1'] = block1
          attention_weights[f'decoder_layer{i+1}_block2'] = block2
      #print('shape', x.shape)
      # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights
#创建transformer#####################################################################3


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  # predictions, attention_weights = transformer(encoder_input,  # (1,25)
  #                                              output,  # (1,1)
  #                                              False,
  #                                              enc_padding_mask,  # enc (1, 1, 1, 25)
  #                                              combined_mask,  # (1, 1, 1, 1)解码器第一个
  #                                              dec_padding_mask)  # (1, 1, 1, 25)解码器第二个
  def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):
    # transformer(encoder_input,output,False,enc_padding_mask,
    #             combined_mask, dec_padding_mask)
    #encoder_input(1,25),False,enc_padding_mask(1, 1, 1, 25)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    #print('enc.out',enc_output.shape)#(32, 25, 128)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    #print('tar_seq_length',seq_length)
    #print('tar',tar.shape)#tar (32,1)
    dec_output, attention_weights = self.decoder(#output,enc_output,False,第一块combined_mask,第二块dec_padding_mask
      tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    #print('dec_output',dec_output.shape)#dec_output (32, 1, 128)
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    #print('final',final_output.shape)#final (32, 1, 19)
    return final_output, attention_weights

num_layers = 8
d_model = 512
dff = 2048
num_heads = 8

input_vocab_size = len(words)+3
target_vocab_size = len(words)+5#tokenizer_en.vocab_size + 2
dropout_rate = 0.1
#优化器###############################################################################
# #将 Adam 优化器与自定义的学习速率调度程序（scheduler）配合使用
#加快学习算法的一个办法就是随时间慢慢减少学习率,我们将之称为学习率衰减(learning rate decay)
# 在训练过程中，我们可以根据训练的结果对学习率做出改变，动态改变学习速率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)#学习率更新

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# temp_learning_rate_schedule = CustomSchedule(d_model)

#损失和指标，交叉熵损失函数##################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(#from_logits: 为True时，还未经过softmax，softmax实现在损失函数中
    from_logits=True, reduction='none')#自定义损失函数，计算批量样本个数的loss的sum总和返回一个总和值
def loss_function(real, pred):
  """自定义损失函数，参数为预测结果pred和真实结果real"""
    # 使用tf.math.equal方法对real和0进行对比
    # 对结果再进行逻辑非操作生成掩码张量mask

  mask = tf.math.logical_not(tf.math.equal(real, 0))#real真实值为0=False real真实值不为0=True
  ## 使用基本计算方法计算损失
  loss_ = loss_object(real, pred)
  # 将mask进行类型转换，使其能够进行后续操作
  mask = tf.cast(mask, dtype=loss_.dtype)
  # 将loss_与mask相乘即对loss_进行掩码
  loss_ *= mask
#    # 计算loss_张量所有元素的均值return tf.reduce_mean(loss_)
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

#计算给定值的（加权）平均值
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=seq_length,#input_vocab_size,
                          pe_target=seq_length,#target_vocab_size,
                          rate=dropout_rate)

#  def call(self, inp, tar, training, enc_padding_mask,look_ahead_mask, dec_padding_mask):
#创建mask##################################################################
def create_masks(inp, tar):
  # Encoder padding mask编码器
  enc_padding_mask = create_padding_mask(inp)#创建输入数据input的padding

  # Used in the 2nd attention block in the decoder.用于解码器的第二个attention
  # This padding mask is used to mask the encoder outputs.用于编码器的输出
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.解码器的第一个attenion，屏蔽未来信息
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = "./checkpoints/zip.8"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 20
#训练步骤######################################################################
train_step_signature = [#input_signature限定输入类型
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=100,
#     target_vocab_size=100,
#     # pe_input=25,
#     # pe_target=25,
#     rate=dropout_rate)
#该函数专用于参数张量的精确形状，指定通用形状
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  # print('inp',inp.shape)
  # print(inp)
  # print('tar',tar.shape)
  # print(tar)
  tar_inp =tar[:, :-1]
  tar_real =tar[:, 1:]
  # print('tar_inp', tar_inp.shape)
  # print(tar_inp)
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    ##  def call(self, inp, tar, training, enc_padding_mask,look_ahead_mask, dec_padding_mask):
    loss = loss_function(tar_real, predictions)
  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss.update_state(loss)
  train_accuracy.update_state(accuracy_function(tar_real, predictions))

fig_loss = np.zeros([1200])
fig_accuracy = np.zeros([1200])
i=-1
# for epoch in range(EPOCHS):
#   start = time.time()
#
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#
#   # inp -> portuguese, tar -> english
#   for batch in range(len(x_batches)):
#   #for (batch, (inp, tar)) in enumerate(train_dataset):
#     # print('batch',batch)
#     #print('x',x_batches[batch])
#     #print('y',y_batches[batch])
#     x_batches[batch]=tf.convert_to_tensor(x_batches[batch],dtype=tf.int64)
#     y_batches[batch] = tf.convert_to_tensor(y_batches[batch],dtype=tf.int64)
#     # print('x', x_batches[batch])
#     # print('y', y_batches[batch])
#     train_step(x_batches[batch],
#                y_batches[batch])
#     i+=1
#     fig_loss[i] =train_loss.result().numpy()
#     fig_accuracy[i] = train_accuracy.result().numpy()
#     if batch % 10 == 0:
#       print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#         epoch + 1, batch, train_loss.result().numpy(), train_accuracy.result().numpy()))
#
#   # if (epoch + 1) % 5 == 0:
#   ckpt_save_path = ckpt_manager.save()
#   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
#                                                       ckpt_save_path))
#
#   print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
#                                                       train_loss.result().numpy(),
#                                                       train_accuracy.result().numpy()))
#
# print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# lns1 = ax1.plot(np.arange(1200), fig_loss,'b', label='Loss')
# lns2 = ax2.plot(np.arange(1200), fig_accuracy, 'r', label='Accuracy')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('training loss')
# ax2.set_ylabel('training accuracy')
# lns = lns1 + lns2
# labels = ['Loss', 'Accuracy']
# plt.legend(lns, labels, loc=7)
# plt.show()
##############################################################################################
#print('测试!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#  def call(self, inp, tar, training, enc_padding_mask,look_ahead_mask, dec_padding_mask):
#
def evaluate(output_file,inp_sentence):

  encoder_input = inp_sentence
  #print('enc', encoder_input)#[7 2 3 1 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
  encoder_input=tf.expand_dims(encoder_input,0)#(1,25)
  # print('enc_inp2',encoder_input)
  # tf.Tensor([[1  2 18 17 14 14 10  4 18 17 18  3 18 16 14  0  0  0  0  0  0  0  0  0
  #     0]], shape=(1, 25), dtype=int64)
  #print('en_inp',encoder_input)#(batch_size,seq_length)[[ 1  2 18 17 14 14 10  4 18 17 18  3 18 16 14  0  0  0  0  0  0  0  0  0 0]]
  # 因为目标是英语，输入 transformer 的第一个词应该是
  # 英语的开始标记。
  decoder_input = [len(words)+3]
  #print('dec_inp',decoder_input)
  output=tf.expand_dims(decoder_input,0)
  ###############output = tf.expand_dims(output, 0)#tensor([1]),shape=(1,)
  ###############print('output1',output)#tf.Tensor([[1]], shape=(1, 1), dtype=int64)
  for i in range(seq_length):
    #print('i',i)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
    # print('enc',enc_padding_mask.shape)#enc (1, 1, 1, 25)
    # print('com',combined_mask.shape)#com (1, 1, 1, 32, 32)已改为(1, 1, 1, 1)
    # print('dec',dec_padding_mask.shape)#dec (1, 1, 1, 25)
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input,#(1,25)
                                                 output,#(1,1)
                                                 False,
                                                 enc_padding_mask,#enc (1, 1, 1, 25)
                                                 combined_mask,#(1, 1, 1, 1)解码器第一个
                                                 dec_padding_mask)#(1, 1, 1, 25)解码器第二个

    # 从 seq_len 维度选择最后一个词??????

    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)(1,1,11)
    prediction=tf.nn.softmax(predictions)
    print('pre',prediction)
    #with open('input_traces1.1/input.txt', "r") as f:
    with open(output_file, 'a') as writer:
      #words [0='<END>',1= '<START>', 2='atwo', 3='bthree', 4='cfour', 5='dfive']
      arr=[(words[j],prediction[:,:,j+1][0][0].numpy()) for j in range(len(words))]
      #arr = sorted(arr, key=lambda x: x[1], reverse=True)  # （方法，概率）按概率排序
      writer.write('1-TAIL\t' + '\t'.join([w + ':' + str(p) for (w, p) in arr]) + '\n')
      # print('i',i)
      # print('inp_sentence[i+1]',inp_sentence[i+1])
      if inp_sentence[i+1]==len(words)+1:
        www='<ISOS>'
        writer.write('WORD\t' + www + '\n')
        writer.close()
      elif inp_sentence[i+1]==len(words)+2:
        www='<IEOS>'
        writer.write('WORD\t' + www + '\n')
        writer.close()
      else:
        www=words[inp_sentence[i+1]-1]
        writer.write('WORD\t' + www + '\n')
        writer.close()
        if www=='<END>':
          return tf.squeeze(output, axis=0), attention_weights
    predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
    print('id',predicted_id)#(32,1)
    output = tf.concat([output, predicted_id], axis=-1)
 # tf.squeeze(output, axis=0)
  return output, attention_weights

def translate(sentence,plot=''):
  index=0
  dir = '/zip.8'
  b = os.getcwd()
  if os.path.isdir(b+dir):
    import shutil
    shutil.rmtree(b+dir)  # 递归地删除文件夹以及里面的文件
  os.makedirs(b + dir)
  for tr in sentence:
    print('tr',tr)
    index+=1
    print('第',index, '条轨迹')
    a=b + dir+'/d'+str(index)+'.txt'
    # result, attention_weights =
    evaluate(a,tr)

    # if plot:
    #    plot_attention_weights(attention_weights, sentence, result, plot)
translate(test)
