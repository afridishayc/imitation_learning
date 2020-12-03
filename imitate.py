import json
import pickle
import numpy as np
from franka import FrankaEnv
from model_def import CombinedModel

# env = FrankaEnv(visual=False)

task_sequence = []

with open("target_task.jsl", "r") as f:
	for line in f.readlines():
		line = np.array(json.loads(line))
		task_sequence.append(line)


combined_model = CombinedModel(16, 8, 4)
combined_model.load_weights(filepath="./combined_checkpoints/mcombined_model_weights5000")

print(task_sequence)


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

env = FrankaEnv(visual=False)
prev_state = env.reset()

for task in task_sequence:
	inp_state1 = np.array(prev_state["end_effector"]).astype(np.float32)
	inp_state2 = np.array(prev_state["info"]).astype(np.float32)
	# , np.array(prev_state["info"], dtype=np.float32))
	inp_state = np.array([np.concatenate((inp_state1, inp_state2, task[0]))])
	# print(inp_state.shape)
	# inp_state = task[0
	test_prev_states, test_actions, test_current_states = datastore.sample(4)
	test_states = np.append(test_prev_states, test_current_states, axis=1)
	predicted_action = combined_model.action_prediction(test_states)
	print(action)	
	print(predicted_action)
	break