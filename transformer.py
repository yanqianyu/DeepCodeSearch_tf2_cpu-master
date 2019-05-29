from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as k
import numpy as np
import tensorflow as tf
import math


# embedding
# input -> Nonelength
class Embedding(Layer):
	def __init__(self, units, vocab_size, name):
		super(Embedding, self).__init__(name=name+'embed')
		self.model_dim = units
		self.embdding = layers.Embedding(input_dim=vocab_size, output_dim=units, mask_zero=False)
		# self.lookup = self.add_weight(name='lookup', shape=[vocab_size, self.model_dim], initializer="random_normal")
		self.scale = math.sqrt(self.model_dim)

	def call(self, inputs):
		embdding = self.embdding(inputs)
		# embdding = tf.nn.embedding_lookup(self.lookup, inputs)
		embdding = embdding * self.scale

		return embdding

# None,length -> None,length,512 / sqr(512)


# position FFN
# output = w2(droput(relu(w1*input)))
# input -> None,length, 512
class positionFeedForward(Layer):
	# just two layer feedforward
	def __init__(self, model_dim, name, ffn_dim=512, droput=0.2):
		super(positionFeedForward, self).__init__(name=name+'FFN')
		self.dense1 = layers.Dense(ffn_dim, 'relu')
		self.dense2 = layers.Dense(model_dim)
		self.dropout = droput

	def call(self, inputs):
		return self.dense2(tf.nn.dropout(self.dense1(inputs), self.dropout))
# None,length,512 -> None,length,512


class positionEmbedding(Layer):
	def __init__(self, model_dim, max_len, name, dropout=0.2):
		super(positionEmbedding, self).__init__(name=name+'poEmbed')
		self.model_dim = model_dim
		self.dropout = dropout
		self.max_len = max_len
		self.pe = np.zeros([max_len, model_dim])
		self.position = np.arange(0, max_len)
		self.position = np.expand_dims(self.position, axis=1)  # max_len,1
		cons = -(math.log(10000.0) / model_dim)
		self.div_term = np.exp(np.arange(0, model_dim, 2) * cons)
		self.pe[:, 0::2] = np.sin(self.position * self.div_term)
		self.pe[:, 1::2] = np.cos(self.position * self.div_term)
		self.pe = np.expand_dims(self.pe, 0)
		self.pe = np.tile(self.pe, [128, 1, 1])
		self.pe = tf.Variable(self.pe, trainable=False, dtype=tf.float32)
		# self.scale = model_dim ** 0.5
		# max_len, model_dim
		# self.pe = np.expand_dims(self.pe, axis=0)

	def call(self, x):
		# out = tf.expand_dims(tf.range(int(x.shape[1])), 0)
		# x = tf.tile(x, [128, 1, 1])
		#
		# out = tf.nn.embedding_lookup(self.pe, posi_ID)
		x = x + self.pe
		# None,length, model_dim
		return tf.nn.dropout(x, 0.1)


# layer normlize
class layerNorm(Layer):
	def __init__(self, name, model_dim):
		super(layerNorm, self).__init__(name=name+'layerNorm')
		self.eps = 1e-6
		self.a = tf.Variable(tf.ones(model_dim))
		self.b = tf.Variable(tf.zeros(model_dim))

	def call(self, x):
		mean = tf.reduce_mean(x, axis=-1, keepdims=True)
		std = tf.math.reduce_std(x, axis=-1, keepdims=True)
		# mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
		# std = tf.sqrt(std + self.eps)
		#  0.1 *
		return 0.5 * self.a * (x - mean) / (std + self.eps) + self.b
		# return (x - mean) / (std + self.eps)


# A encoder layer
# outputs = feed_forward(self_attention(inputs))
# shape have no changed!
class encoder_layer(Layer):
	def __init__(self, hidden, self_atten, feed_forward, dropout, name):
		super(encoder_layer, self).__init__(name=name+'encoder_layer')
		self.self_atten = self_atten
		self.hidden = hidden
		self.feed_forward = feed_forward
		self.norm = layerNorm(name, hidden)
		self.dropout = dropout

	def call(self, x, mask):
		# self_attention
		atten_output = self.self_atten(self.norm(x), mask=mask)
		# add & norm
		output_1 = tf.nn.dropout(atten_output, self.dropout) + x
		# going to feed forward
		output_2 = tf.nn.dropout(self.feed_forward(self.norm(output_1)), self.dropout) + output_1
		# add & norm
		return output_2


# stack encoder_layer to the encoder
class encoder(Layer):
	def __init__(self, hidden, self_atten, feed_forward, name, dropout=0.2, n_layers=8):
		super(encoder, self).__init__(name=name+'encoder')
		self.encoder_layers = [encoder_layer(hidden, self_atten, feed_forward, dropout, name=str(i)) for i in range(n_layers)]
		self.n_layers = n_layers
		self.norm = layerNorm(name, hidden)

	def call(self, x, mask):
		self.out = x
		for i in range(self.n_layers):
			self.out = self.encoder_layers[i](self.out, mask=mask)

		return self.norm(self.out)


# muti_head self_attention
# q == k == v == inputs -> dim does'nt changed
class self_atten(Layer):
	def __init__(self, n_heads, model_dim, name, dropout=0.1):
		super(self_atten, self).__init__(name=name+'atten')
		# Make sure that inputs can split to n_heads*batch, model_dim/n_heads successful.
		assert model_dim % n_heads == 0
		self.head_dim = model_dim // n_heads
		self.linears = [layers.Dense(model_dim, activation='relu') for _ in range(4)]
		self.n_heads = n_heads
		self.model_dim = model_dim
		self.dropout = dropout

	def attention(self, query, key, value, dropout=0.1, src_mask=None):
		head_dim = query.shape[-1]
		# q * k & scale(sqr(dim)) -> None,length, n_heads, dim/n_heads,
		term = tf.constant(math.sqrt(int(head_dim)), dtype=tf.float32)

		key = tf.transpose(key, [0, 1, 3, 2])
		key = key / term
		atten_weight = tf.matmul(query, key)
		# None, n_heads, length, length
		if src_mask is not None:
			mask = tf.expand_dims(src_mask, 1)
			mask = tf.tile(mask, [1, self.n_heads, 1, 1])
			# MIN_VALUE = tf.constant(-1e9, shape=(query.shape[0], atten_weight.shape[1:]))
			MIN_VALUE = tf.fill(atten_weight.shape, value=-1e9)
			atten_weight = tf.where(tf.equal(mask, 1), atten_weight, MIN_VALUE)

		output = tf.nn.softmax(atten_weight, axis=-1)
		output = tf.matmul(output, value)  # None, n_heads, length, dim/n_heads
		return output
		# None,length, n_heads, dim/n_heads

	def call(self, query, mask):
		# split heads
		query = self.linears[0](query)
		key = self.linears[1](query)
		value = self.linears[2](query)
		# None,length, 512 -> None,length, n_heads, dim/n_heads -> None, n_heads,length, dim/n_heads
		query = tf.transpose(tf.reshape(query, [-1, query.shape[1], self.n_heads, self.head_dim]), (0, 2, 1, 3))
		key = tf.transpose(tf.reshape(key, [-1, key.shape[1], self.n_heads, self.head_dim]), (0, 2, 1, 3))
		value = tf.transpose(tf.reshape(value, [-1, value.shape[1], self.n_heads, self.head_dim]), (0, 2, 1, 3))
		output = self.attention(query, key, value, self.dropout, src_mask=mask)
		# concat to get output like None,length, model_dim
		output = tf.reshape(output, (-1, query.shape[2], self.model_dim))
		# print(output.shape)
		return self.linears[-1](output)
		# None,length, model_dim


def getMask(inputs):
	# 这里输入的还是np矩阵
	src_mask = k.not_equal(inputs, 0)  # 0->False
	src_mask = k.expand_dims(src_mask, 1)  # None, 1, length
	# self.src_mask = tf.tile(self.src_mask, [1, src.shape[1], 1])  # None, length, length
	src_mask2 = tf.transpose(src_mask, [0, 2, 1])  # None, 1, length
	src_mask = src_mask & src_mask2  # None, length, length
	src_mask = tf.cast(src_mask, dtype=tf.float32)
	# self.src_mask = tf.reshape(self.src_mask, [-1, src.shape[1], src.shape[1]])  # None, length, length,
	return src_mask


class EncoderModel(Layer):
	def __init__(self, model_dim, embed_dim, vocab_size, n_heads, ffn_dim, droput_rate, max_len, name):
		super(EncoderModel, self).__init__(name=name)
		self.model_dim = model_dim
		self.embed_dim = embed_dim
		self.vocab_size = vocab_size
		self.n_heads = n_heads
		self.ffn_dim = ffn_dim
		self.droput_rate = droput_rate
		print('init the '+name)
		self.attention = self_atten(n_heads, model_dim, name, droput_rate)
		self.feedForward = positionFeedForward(model_dim, name, ffn_dim)
		self.vocab_embed = Embedding(embed_dim, vocab_size, name)
		self.position_embed = positionEmbedding(embed_dim, max_len, name, droput_rate)
		self.encoder = encoder(model_dim, self.attention, self.feedForward, name, droput_rate, n_layers=4)

	def call(self, inputs):
		vocab_embed = self.vocab_embed(inputs)
		position_embed = self.position_embed(vocab_embed)
		outputs = self.encoder(position_embed, mask=getMask(inputs))
		return outputs

