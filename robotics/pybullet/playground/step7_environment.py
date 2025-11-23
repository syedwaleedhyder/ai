import pybullet as p
import pybullet_data
import time

# 1. Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 2. Load Ground
planeId = p.loadURDF("plane.urdf")

# 3. Load Sawyer
# We place the robot at Z = 0.9 (approx 3 feet in the air)
robotStartPos = [0, 0, 0.9]
robotStartOr = p.getQuaternionFromEuler([0, 0, 0])
sawyerId = p.loadURDF("./sawyer_robot/sawyer_description/urdf/sawyer.urdf", robotStartPos, robotStartOr, useFixedBase=1)

# 4. Load Table
# We place it slightly in front of the robot
tablePos = [0.8, 0, 0.0] 
# Scaling it down slightly to make it more "workbench" sized
tableId = p.loadURDF("table/table.urdf", tablePos, useFixedBase=1, globalScaling=0.8)

# 5. Load Cube
# The table surface height changes because of scaling.
# A standard table is ~0.7m. Scaled 0.8x is ~0.56m.
# But remember, PyBullet tables are weird. Let's drop it from high up to be safe.
cubeStartPos = [0.7, 0, 1.0] 
cubeId = p.loadURDF("cube.urdf", cubeStartPos, globalScaling=0.08) # Smaller cube (8cm)

# 6. Physics settle
for _ in range(100):
    p.stepSimulation()

# 7. Reset Camera to look at the table surface
p.resetDebugVisualizerCamera( cameraDistance=1.2, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[0.4, 0, 1.0])

while True:
    p.stepSimulation()
    time.sleep(1./240.)