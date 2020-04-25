import numpy as np
import json	
from franka import FrankaEnv	

env = FrankaEnv(visual=False)
prev_state = env.reset()

i = 0
with open("agent_interactions.jsl", "w+") as f:
	for p in range(200):
		prev_state = env.reset()
		for t in range(500):
			# env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			record = np.array([np.asarray(prev_state["end_effector"]), action, np.asarray(observation["end_effector"]), done])
			print(record)
			record = np.array2string(record)
			f.write(json.dumps({
									"prev_state": np.array2string(np.asarray(prev_state["end_effector"]), separator=',').replace("\n", ""),
									"action": np.array2string(action, separator=',').replace("\n", ""),
									"current_state": np.array2string(np.asarray(observation["end_effector"]), separator=',').replace("\n", ""),
									"done": done,
									"info": np.array2string(np.asarray(info), separator=',').replace("\n", "")
								}) + "\n")
			# np.savetxt(f, record, delimiter=' ', newline=",", fmt='%s')
			prev_state = observation
			i += 1
			print(i)
env.close()

