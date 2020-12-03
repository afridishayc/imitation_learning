import pickle
import tensorflow as tf
import numpy as np
from datastore import DataStore
from model_def import CombinedModel
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


combined_model = CombinedModel(16, 8, 4)
optimizer = Adam(learning_rate=0.00001)


@tf.function
def train_step(train_states, prev_states, current_states, actions):
	with tf.GradientTape() as tape:
		predicted_states = combined_model(train_states, training=True)
		predicted_action = combined_model.action_prediction(train_states)
		predicted_action_real_action = combined_model.state_prediction(actions)
		loss = combined_model.get_loss(prev_states, current_states, actions, predicted_action_real_action, predicted_states, predicted_action, lambda_=1)
	gradients = tape.gradient(loss, combined_model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, combined_model.trainable_variables))
	tf.print(loss)


# combined_model.load_weights(filepath="./combined_checkpoints/mcombined_model_weights130000")

for i in range(10000):
	prev_states, actions, current_states = datastore.sample(16)
	train_states = np.append(prev_states, current_states, axis=1)
	train_step(train_states, prev_states, current_states, actions)
	# print(i)
	if i % 5000 == 0:
		test_prev_states, test_actions, test_current_states = datastore.sample(4)
		test_states = np.append(test_prev_states, test_current_states, axis=1)
		# print(test_states.shape)
		# print(test_actions.shape)
		# print(combined_model(test_states))
		# print(test_actions)
		combined_model.save_weights("combined_checkpoints/mcombined_model_weights" + str(i))

# combined_model.save_weights("combined_checkpoints/mcombined_model_weights" + str(i))