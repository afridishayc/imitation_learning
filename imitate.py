import json
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
combined_model.load_weights(filepath="./combined_checkpoints/mcombined_model_weights795000")

print(task_sequence)


env = FrankaEnv(visual=False)
prev_state = env.reset()

for task in task_sequence:
	inp_state1 = np.array(prev_state["end_effector"]).astype(np.float32)
	inp_state2 = np.array(prev_state["info"]).astype(np.float32)
	# , np.array(prev_state["info"], dtype=np.float32))
	inp_state = np.array([np.concatenate((inp_state1, inp_state2, task[0]))])
	print(inp_state.shape)
	# inp_state = task[0]
	predicted_action = combined_model.action_prediction(inp_state)	
	# print(predicted_action)