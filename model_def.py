import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class ActionModel(Model):

	def __init__(self, input_size, output_size):
		super(ActionModel, self).__init__()
		self.inp_l = Dense(512, input_shape=(input_size,))
		self.l1 = Dense(512, activation='relu')
		self.l2 = Dense(512, activation='relu')
		self.l3 = Dense(512, activation='relu')
		self.l4 = Dense(512, activation='relu')
		# self.l5 = Dense(512, activation='relu')
		# self.l6 = Dense(512, activation='relu')
		# self.l7 = Dense(512, activation='relu')
		self.out_l = Dense(output_size, activation='tanh')

	def call(self, inp_data):
		output = self.inp_l(inp_data)
		output = self.l1(output)
		output = self.l2(output)
		output = self.l3(output)
		output = self.l4(output)
		# output = self.l5(output)
		# output = self.l6(output)
		# output = self.l7(output)
		output = self.out_l(output)
		return output

	def get_loss(self, y_real, y_pred):
		# tf.print(y_real)
		# tf.print(y_pred)
		# tf.print(tf.math.square(y_real - y_pred))
		return tf.reduce_sum(tf.math.square(y_real - y_pred), axis=1)


class ConsistencyModel(Model):

	def __init__(self, input_size, output_size):
		super(ConsistencyModel, self).__init__()
		self.inp_l = Dense(128, input_shape=(input_size,))
		self.l1 = Dense(128, activation='relu')
		self.l2 = Dense(128, activation='relu')
		self.l3 = Dense(128, activation='relu')
		self.l4 = Dense(64, activation='relu')
		self.out_l = Dense(output_size, activation='linear')

	def call(self, inp_data):
		output = self.inp_l(inp_data)
		output = self.l1(output)
		output = self.l2(output)
		output = self.l3(output)
		output = self.l4(output)
		output = self.out_l(output)
		return output

	def get_loss(self, y_real, y_pred):
		return tf.reduce_sum(tf.math.square(y_real - y_pred), axis=1)


class CombinedModel(Model):

	def __init__(self, input_size, output_size, mid_layer_size):
		super(CombinedModel, self).__init__()
		self.inp_l = Dense(512, input_shape=(input_size,))
		self.l1 = Dense(512, activation='relu')
		self.l2 = Dense(512, activation='relu')
		self.l3 = Dense(512, activation='relu')
		self.l4 = Dense(512, activation='relu')
		self.actions_layer = Dense(mid_layer_size, activation='tanh')
		self.f1 = Dense(128, input_shape=(mid_layer_size,))
		self.f2 = Dense(128, activation='relu')
		self.f3 = Dense(128, activation='relu')
		self.f4 = Dense(128, activation='relu')
		self.f5 = Dense(64, activation='relu')
		self.out_l = Dense(output_size, activation='linear')

	def call(self, inp_data):
		output = self.inp_l(inp_data)
		output = self.l1(output)
		output = self.l2(output)
		output = self.l3(output)
		output = self.l4(output)
		output = self.actions_layer(output)
		output = self.f1(output)
		output = self.f2(output)
		output = self.f3(output)
		output = self.f4(output)
		output = self.f5(output)
		output = self.out_l(output)
		return output

	def action_prediction(self, inp_data):
		output = self.inp_l(inp_data)
		output = self.l1(output)
		output = self.l2(output)
		output = self.l3(output)
		output = self.l4(output)
		output = self.actions_layer(output)
		return output

	def state_prediction(self, inp_data):
		output = self.f1(inp_data)
		output = self.f2(output)
		output = self.f3(output)
		output = self.f4(output)
		output = self.f5(output)
		output = self.out_l(output)
		return output

	def get_loss(self, prev_state, current_state, actions, pred_states_real_action, pred_states_pred_action, pred_action, lambda_=1):
		# tf.print(y_real)
		# tf.print(y_pred)
		# tf.print(tf.math.square(y_real - y_pred))
		# tf.print(tf.reduce_sum(tf.math.square(y_real - y_pred), axis=1))
		return tf.reduce_mean(lambda_ * tf.reduce_sum(tf.math.square(pred_action-actions), axis=1) + tf.reduce_sum(tf.math.square(current_state -pred_states_real_action), axis=1) + tf.reduce_sum(tf.math.square(current_state-pred_states_pred_action), axis=1)) 
