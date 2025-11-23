import pybullet as p
import pybullet_data
import time

# 1. Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# 2. Load Sawyer
# We use a slightly higher start pos so the hand doesn't hit the floor immediately
sawyerId = p.loadURDF("./sawyer_robot/sawyer_description/urdf/sawyer.urdf", [0,0,0], useFixedBase=1)

# 3. Identify the AR10 Hand Motors
# Based on your URDF, these are the main knuckle joints for the 5 fingers.
# servo2 = Index, servo4 = Middle, servo6 = Ring, servo8 = Pinky, servo0 = Thumb
hand_indices = []

num_joints = p.getNumJoints(sawyerId)

print("Mapping AR10 Hand Joints...")
for i in range(num_joints):
    joint_info = p.getJointInfo(sawyerId, i)
    joint_name = joint_info[1].decode("utf-8")
    
    if "servo" in joint_name:
        print(f"Found Hand Joint: {joint_name} (ID: {i})")
        hand_indices.append(i)

# 4. Create Slider
# The URDF limits are approx 0.17 to 1.57 radians. 
# 0.0 = Open Hand, 1.5 = Closed Fist
hand_slider = p.addUserDebugParameter("Close Hand", 0.0, 1.5, 0.0)

# 5. Control Loop
while True:
    # Get slider value (0 to 1.5)
    grip_pos = p.readUserDebugParameter(hand_slider)
    
    # Lift the arm so you can see the hand
    p.setJointMotorControl2(sawyerId, 3, p.POSITION_CONTROL, 0.0) # Shoulder
    p.setJointMotorControl2(sawyerId, 8, p.POSITION_CONTROL, -1.0) # Elbow
    p.setJointMotorControl2(sawyerId, 13, p.POSITION_CONTROL, 1.5) # Wrist
    # Rotate the wrist so palm faces up/out
    p.setJointMotorControl2(sawyerId, 16, p.POSITION_CONTROL, 1.5) 

    # Close the fingers!
    p.setJointMotorControlArray(
        sawyerId, 
        hand_indices, 
        p.POSITION_CONTROL, 
        targetPositions=[grip_pos] * len(hand_indices),
        forces=[5] * len(hand_indices) 
    )

    p.stepSimulation()
    time.sleep(1./240.)