
import os
import gym
import pybullet as pb
import pybullet_data as pbd
import numpy as np
import random
import math

from gym import error, spaces, utils
from gym.utils import seeding



class FrankaEnv(gym.Env):

	def __init__(self, screen_width=720, screen_height=720, visual=True):
		pb.connect(pb.GUI)
		pb.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
		self.observation_space = spaces.Dict({
												"rgba": spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 4)),
												"depth_buffer": spaces.Box(low=0, high=255, shape=(screen_height, screen_width)),
												"end_effector": spaces.Box(np.array([-1]*5), np.array([1]*5))
												})
		self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
		self.is_visual_output = visual

	def reset(self):
		pb.resetSimulation()
		pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,0)
		pb.setGravity(0,0,-9.8)
		urdfRootPath = pbd.getDataPath()
		planeUid = pb.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
		rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
		self.pandaUid = pb.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
		for i in range(7):
			pb.resetJointState(self.pandaUid,i, rest_poses[i])
		tableUid = pb.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
		# trayUid = pb.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
		state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0]
		self.objectUid = pb.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
		state_robot = pb.getLinkState(self.pandaUid, 11)[0]
		state_fingers = (pb.getJointState(self.pandaUid,9)[0], pb.getJointState(self.pandaUid, 10)[0])
		end_effector = state_robot + state_fingers
		pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,1)
		if self.is_visual_output:
			h, w, rgba, depth, mask = pb.getCameraImage(720, 720)
		else:
			rgba = -1
			depth = -1
		return {"rgba": rgba, "depth": depth, "end_effector": end_effector, "info": state_object}

	def step(self, action):
		pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
		orientation = pb.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
		dv = 0.05
		dx = action[0] * dv
		dy = action[1] * dv
		dz = action[2] * dv
		fingers = action[3]

		currentPose = pb.getLinkState(self.pandaUid, 11)
		currentPosition = currentPose[0]
		newPosition = [currentPosition[0] + dx,
						currentPosition[1] + dy,
						currentPosition[2] + dz]
		jointPoses = pb.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

		pb.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], pb.POSITION_CONTROL, list(jointPoses)+2*[fingers])

		pb.stepSimulation()

		state_object, _ = pb.getBasePositionAndOrientation(self.objectUid)
		state_robot = pb.getLinkState(self.pandaUid, 11)[0]
		state_fingers = (pb.getJointState(self.pandaUid,9)[0], pb.getJointState(self.pandaUid, 10)[0])
		if state_object[2]>0.45:
			reward = 1
			done = True
		else:
			reward = 0
			done = False
		info = state_object
		end_effector = state_robot + state_fingers
		if self.is_visual_output:
			h, w, rgba, depth, mask = pb.getCameraImage(720, 720)
		else:
			rgba = -1
			depth = -1
		return {"rgba": rgba, "depth": depth, "end_effector": end_effector}, reward, done, info

	def render(self):
		view_matrix = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
															distance=.7,
															yaw=90,
															pitch=-70,
															roll=0,
															upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(fov=60,
													aspect=float(960) /720,
													nearVal=0.1,
													farVal=100.0)
		(_, _, px, _, _) = pb.getCameraImage(width=960,
												height=720,
												viewMatrix=view_matrix,
												projectionMatrix=proj_matrix,
												renderer=pb.ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (720,960, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
		pb.disconnect()

	def go_to_point(self, coordinates):
		pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
		orientation = pb.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
		currentPose = pb.getLinkState(self.pandaUid, 11)
		currentPosition = currentPose[0]
		jointPoses = pb.calculateInverseKinematics(self.pandaUid,11, coordinates, orientation)[0:7]
		pb.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], pb.POSITION_CONTROL, list(jointPoses)+2*[0])
		pb.stepSimulation()
		state_object, _ = pb.getBasePositionAndOrientation(self.objectUid)
		state_robot = pb.getLinkState(self.pandaUid, 11)[0]
		state_fingers = (pb.getJointState(self.pandaUid,9)[0], pb.getJointState(self.pandaUid, 10)[0])