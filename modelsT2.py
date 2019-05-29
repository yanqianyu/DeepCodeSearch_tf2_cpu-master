import h5py
import numpy as np
from tensorflow.python.keras.layers import Concatenate, Dot, Lambda, Activation, Embedding, Dense, Dropout
from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine import Input
from tensorflow.python.keras import backend as k
from tensorflow.python.keras import layers
from tensorflow.python import keras
import tensorflow as tf
import math
import os
import transformer


class JointEmbeddingModel:
	def __init__(self, config):
		self.data_dir = config.data_dir
		self.model_name = config.model_name
		self.meth_name_len = config.methname_len  # the max length of method name
		self.apiseq_len = config.apiseq_len
		self.tokens_len = config.tokens_len
		self.desc_len = config.desc_len

		self.vocab_size = config.n_words  # the size of vocab
		self.embed_dims = config.embed_dims
		self.lstm_dims = config.lstm_dims
		self.hidden_dims = config.hidden_dims

		self.margin = 0.05

		self.init_embed_weights_meth_name = config.init_embed_weights_methodname
		self.init_embed_weights_tokens = config.init_embed_weights_tokens
		self.init_embed_weights_desc = config.init_embed_weights_desc

		self.meth_name = Input(shape=(self.meth_name_len,), dtype='int32', name='meth_name')
		self.apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		self.tokens = Input(shape=(self.tokens_len,), dtype='int32', name='tokens2')
		self.desc_good = Input(shape=(self.desc_len,), dtype='int32', name='desc_good')
		self.desc_bad = Input(shape=(self.desc_len,), dtype='int32', name='desc_bad')

		if not os.path.exists(self.data_dir + 'model/' + self.model_name):
			os.makedirs(self.data_dir + 'model/' + self.model_name)

	def build(self):

		self.transformer_meth = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.hidden_dims,
		                                                 embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                 droput_rate=0.2, n_heads=8, max_len=self.meth_name_len,
		                                                 name='methT')

		self.transformer_apiseq = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.hidden_dims,
		                                                   embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                   droput_rate=0.2, n_heads=8, max_len=self.apiseq_len,
		                                                   name='apiseqT')

		self.transformer_desc = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.hidden_dims,
		                                                 embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                 droput_rate=0.2, n_heads=8, max_len=self.desc_len, name='descT')

		# self.transformer_ast = EncoderModel(vocab_size=self.vocab_size, model_dim=self.hidden_dims, embed_dim=self.embed_dims, ffn_dim=self.lstm_dims, droput_rate=0.2, n_heads=4, max_len=128)
		self.transformer_tokens = transformer.EncoderModel(vocab_size=self.vocab_size, model_dim=self.hidden_dims,
		                                                   embed_dim=self.embed_dims, ffn_dim=self.lstm_dims,
		                                                   droput_rate=0.2, n_heads=8, max_len=self.tokens_len,
		                                                   name='tokensT')
		# create path to store model Info

		# 1 -- CodeNN
		meth_name = Input(shape=(self.meth_name_len,), dtype='int32', name='meth_name')
		apiseq = Input(shape=(self.apiseq_len,), dtype='int32', name='apiseq')
		tokens3 = Input(shape=(self.tokens_len,), dtype='int32', name='tokens3')

		# method name
		# embedding layer

		meth_name_out = self.transformer_meth(meth_name)
		# max pooling
		maxpool = Lambda(lambda x: k.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_methodname')
		method_name_pool = maxpool(meth_name_out)
		activation = Activation('tanh', name='active_method_name')
		method_name_repr = activation(method_name_pool)

		# apiseq
		# embedding layer

		apiseq_out = self.transformer_apiseq(apiseq)
		# max pooling
		maxpool = Lambda(lambda x: k.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_apiseq')
		apiseq_pool = maxpool(apiseq_out)
		activation = Activation('tanh', name='active_apiseq')
		apiseq_repr = activation(apiseq_pool)

		# tokens
		# embedding layer
		init_emd_weights = np.load(
			self.data_dir + self.init_embed_weights_tokens) if self.init_embed_weights_tokens is not None else None
		init_emd_weights = init_emd_weights if init_emd_weights is None else [init_emd_weights]

		embedding = Embedding(
			input_dim=self.vocab_size,
			output_dim=self.embed_dims,
			weights=init_emd_weights,
			mask_zero=False,
			name='embedding_tokens'
		)
		tokens_embedding = embedding(tokens3)
		# dropout
		dropout = Dropout(0.25, name='dropout_tokens_embed')
		tokens_dropout = dropout(tokens_embedding)

		# forward rnn
		fw_rnn = LSTM(self.lstm_dims, return_sequences=True, name='lstm_tokens_fw')

		# backward rnn
		bw_rnn = LSTM(self.lstm_dims, return_sequences=True, go_backwards=True, name='lstm_tokens_bw')

		tokens_fw = fw_rnn(tokens_dropout)
		tokens_bw = bw_rnn(tokens_dropout)

		dropout = Dropout(0.25, name='dropout_tokens_rnn')
		tokens_fw_dropout = dropout(tokens_fw)
		tokens_bw_dropout = dropout(tokens_bw)

		# max pooling
		maxpool = Lambda(lambda x: k.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_tokens')
		tokens_pool = Concatenate(name='concat_tokens_lstm')([maxpool(tokens_fw_dropout), maxpool(tokens_bw_dropout)])
		tokens_pool = maxpool(tokens_dropout)
		activation = Activation('tanh', name='active_tokens')
		tokens_repr = activation(tokens_pool)
		tokens_repr = tf.reshape(tokens_repr, [128, 256])
		# fusion method_name, apiseq, tokens
		merge_method_name_api = Concatenate(name='merge_methname_api')([method_name_repr, apiseq_repr])
		merge_code_repr = Concatenate(name='merge_code_repr')([merge_method_name_api, tokens_repr])
		print(merge_code_repr)
		code_repr = Dense(self.hidden_dims, activation='tanh', name='dense_coderepr')(merge_code_repr)

		self.code_repr_model = Model(inputs=[meth_name, apiseq, tokens3], outputs=[code_repr], name='code_repr_model')
		self.code_repr_model.summary()

		# self.output = Model(inputs=self.code_repr_model.input, outputs=self.code_repr_model.get_layer('tokensT').output)
		# self.output.summary()

		#  2 -- description
		desc = Input(shape=(self.desc_len,), dtype='int32', name='desc')

		# desc
		# embedding layer
		desc_out = self.transformer_desc(desc)

		# max pooling

		maxpool = Lambda(lambda x: k.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
		                 name='maxpooling_desc')
		desc_pool = maxpool(desc_out)
		activation = Activation('tanh', name='active_desc')
		desc_repr = activation(desc_pool)

		self.desc_repr_model = Model(inputs=[desc], outputs=[desc_repr], name='desc_repr_model')
		self.desc_repr_model.summary()

		#  3 -- cosine similarity
		code_repr = self.code_repr_model([meth_name, apiseq, tokens3])

		desc_repr = self.desc_repr_model([desc])

		cos_sim = Dot(axes=1, normalize=True, name='cos_sim')([code_repr, desc_repr])

		sim_model = Model(inputs=[meth_name, apiseq, tokens3, desc], outputs=[cos_sim], name='sim_model')
		self.sim_model = sim_model

		self.sim_model.summary()

		#  4 -- build training model
		good_sim = sim_model([self.meth_name, self.apiseq, self.tokens, self.desc_good])
		bad_sim = sim_model([self.meth_name, self.apiseq, self.tokens, self.desc_bad])
		loss = Lambda(lambda x: k.maximum(1e-6, self.margin - (x[0] - x[1])), output_shape=lambda x: x[0], name='loss')(
			[good_sim, bad_sim])

		self.training_model = Model(inputs=[self.meth_name, self.apiseq, self.tokens, self.desc_good, self.desc_bad],
		                            outputs=[loss], name='training_model')

		self.training_model.summary()

	def compile(self, optimizer, **kwargs):
		optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
		# optimizer = keras.optimizers.Adam(lr=0.0001)
		# print(self.code_repr_model.layers, self.desc_repr_model.layers, self.training_model.layers, self.sim_model.layers)
		self.code_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.desc_repr_model.compile(loss='cosine_proximity', optimizer=optimizer, **kwargs)
		self.training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)
		self.sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

	def fit(self, x, **kwargs):
		y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
		return self.training_model.fit(x, y, **kwargs)

	def getOutput(self, x):
		# functor = k.function([self.code_repr_model.layers[0].input, k.learning_phase()], [self.code_repr_model.layers[0].output])
		# print(functor(x)[0])
		print(self.output.predict(x))

	def repr_code(self, x, **kwargs):
		return self.code_repr_model.predict(x, **kwargs)

	def repr_desc(self, x, **kwargs):
		return self.desc_repr_model.predict(x, **kwargs)

	def predict(self, x, **kwargs):
		return self.sim_model.predict(x, **kwargs)

	def save(self, code_model_file, desc_model_file, **kwargs):
		file = h5py.File(code_model_file, 'w')
		weight_code = self.code_repr_model.get_weights()
		for i in range(len(weight_code)):
			file.create_dataset('weight_code'+str(i), data=weight_code[i])
		file.close()

		file = h5py.File(desc_model_file, 'w')
		weight_desc = self.desc_repr_model.get_weights()
		for i in range(len(weight_desc)):
			file.create_dataset('weight_desc'+str(i), data=weight_desc[i])
		file.close()
		# self.code_repr_model.save_weights(code_model_file, **kwargs)
		# self.desc_repr_model.save_weights(desc_model_file, **kwargs)

	def load(self, code_model_file, desc_model_file, **kwargs):
		# self.code_repr_model.load_weights(code_model_file, **kwargs)
		# self.desc_repr_model.load_weights(desc_model_file, **kwargs)
		file = h5py.File(code_model_file, 'r')
		weight_code = []
		for i in range(len(file.keys())):
			weight_code.append(file['weight_code'+str(i)][:])
		self.code_repr_model.set_weights(weight_code)
		file.close()

		file = h5py.File(desc_model_file, 'r')
		weight_desc = []
		for i in range(len(file.keys())):
			weight_desc.append(file['weight_desc'+str(i)][:])
		self.desc_repr_model.set_weights(weight_desc)
		file.close()



