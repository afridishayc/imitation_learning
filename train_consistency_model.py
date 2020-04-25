import pickle
import tensorflow as tf
import numpy as np
from datastore import DataStore
from model_def import ConsistencyModel
from tensorflow.keras.optimizers import Adam


# pass the mode OLD to read the picle datastore object
MODE = "OLD"

if MODE == "NEW":
	datastore = DataStore()
	# interactions file
	datastore.load_data("agent_interactions.jsl")
	datastore.build()
	with open('dataset_pick.pkl', 'wb') as o:
		pickle.dump(datastore, o, pickle.HIGHEST_PROTOCOL)

else:
	with open("dataset_pick.pkl", "rb") as p:
		datastore = pickle.load(p)

print(datastore.actions.shape)


consistency_model = ConsistencyModel(12, 8)
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = Adam(learning_rate=0.00001)

@tf.function
def train_step(states, true_states):
	with tf.GradientTape() as tape:
		predicted_states = consistency_model(states, training=True)
		loss = loss_object(true_states, predicted_states)
	gradients = tape.gradient(loss, consistency_model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, consistency_model.trainable_variables))
	tf.print(loss)


# consistency_model.load_weights(filepath="./individual_checkpoints/consistency_model_weights400000")

for i in range(1000000):
	prev_states, actions, current_states = datastore.sample(16)
	train_states = np.append(prev_states, actions, axis=1)
	train_step(train_states, current_states)
	# print(i)
	if i % 5000 == 0:
		test_prev_states, test_actions, test_current_states = datastore.sample(4)
		test_states = np.append(test_prev_states, test_actions, axis=1)
		# print(test_states.shape)
		# print(test_actions.shape)
		print(consistency_model(test_states))
		print(test_current_states)
		consistency_model.save_weights("individual_checkpoints/mconsistency_model_weights" + str(i))

consistency_model.save_weights("individual_checkpoints/mconsistency_model_weights" + str(i))