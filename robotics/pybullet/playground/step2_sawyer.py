import pybullet as p
import pybullet_data
import time
import os

# 1. Setup
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 2. Load Ground
planeId = p.loadURDF("plane.urdf")

# 3. Define the path to the Sawyer URDF
# This assumes you put the sawyer_robot folder next to this script.
sawyer_urdf_path = "/Users/waleed/Documents/Waleed/WSU/Fall25/Robotics/assg/learn/sawyer_robot/sawyer_description/urdf/sawyer.urdf"

# 4. Load Sawyer
# useFixedBase=1 tells PyBullet to weld the robot's base to the world at (0,0,0).
# startPos moves the robot slightly up so it sits ON the floor, not IN it.
startPos = [0, 0, 0.9] # The table height in many sims, or 0 if on floor. Let's try 0.
startPos = [0, 0, 0] 
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

try:
    sawyerId = p.loadURDF(sawyer_urdf_path, startPos, startOrientation, useFixedBase=1)
    print("Sawyer loaded successfully!")
except Exception as e:
    print(f"Error loading Sawyer! Check your path: {sawyer_urdf_path}")
    print(e)

# 5. Simulation Loop
while True:
    p.stepSimulation()
    time.sleep(1./240.)