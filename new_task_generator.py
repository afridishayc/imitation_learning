import os
import pybullet as p
import pybullet_data
import math
import json
import numpy as np

p.connect(p.GUI)
urdfRootPath=pybullet_data.getDataPath()
pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
p.setGravity(0,0,-9.8)
objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
state_durations = [1,1,1,1]
control_dt = 1./240.
p.setTimestep = control_dt
state_t = 0.
current_state = 0
with open("target_task.jsl", "w") as f:
    while True:
        state_t += control_dt
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        currentPose = p.getLinkState(pandaUid, 11)
        currentPosition = currentPose[0]
        
        if current_state == 0:
            jointPoses = p.calculateInverseKinematics(pandaUid,11, [0.5,0,1], orientation)[0:7]
            p.setJointMotorControlArray(pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[0])

        if current_state == 1:
            jointPoses = p.calculateInverseKinematics(pandaUid,11, [0.2,0.2,0.8], orientation)[0:7]
            p.setJointMotorControlArray(pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[0])

        if current_state == 2:
            jointPoses = p.calculateInverseKinematics(pandaUid,11, [0.5,0.6,0.8], orientation)[0:7]
            p.setJointMotorControlArray(pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[0])
        
        if current_state == 3:
            jointPoses = p.calculateInverseKinematics(pandaUid,11, [0.5,0,1], orientation)[0:7]
            p.setJointMotorControlArray(pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[0])

        
        state_robot = p.getLinkState(pandaUid, 11)[0]
        state_fingers = (p.getJointState(pandaUid,9)[0], p.getJointState(pandaUid, 10)[0])
        end_effector = state_robot + state_fingers
        state_object, _ = p.getBasePositionAndOrientation(objectUid)
        dp = np.asarray(end_effector + state_object).tolist()
        f.write(json.dumps([dp]) + "\n")
        
        if state_t > state_durations[current_state]:
            current_state += 1
            if current_state >= len(state_durations):
                break
            state_t = 0
        p.stepSimulation()

        
