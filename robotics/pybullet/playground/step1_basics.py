import pybullet as p
import pybullet_data
import time

# 1. Connect to the Physics Server
# p.GUI will open the graphical window. 
# p.DIRECT would run it without a window (faster, used for training AI later).
physicsClient = p.connect(p.GUI)

# 2. Add the path to basic assets (like the floor)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Load the Ground
# The function returns a unique ID for the object, which we can use to track it later.
planeId = p.loadURDF("plane.urdf")

# 4. Set Gravity
# PyBullet uses (X, Y, Z). We set Z to -9.8 (Earth's gravity downwards).
p.setGravity(0, 0, -9.8)

# 5. Load a Cube
# We spawn it at position [0, 0, 1] (1 meter in the air) so it falls.
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0., 0.785398, 0])
boxId = p.loadURDF("cube.urdf", startPos, startOrientation)

# 6. The Simulation Loop
print("Starting simulation... Close the window to stop.")
while True:
    p.stepSimulation()
    # We sleep for 1/240 of a second to match PyBullet's default time step
    # Otherwise, the simulation runs as fast as your CPU allows (too fast to see!)
    time.sleep(1./240.)