import pybullet as p
import pybullet_data
import time
import math

# --- SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 1. Load Environment
p.loadURDF("plane.urdf")
sawyerId = p.loadURDF("../sawyer_robot/sawyer_description/urdf/sawyer.urdf", [0, 0, 0.9], useFixedBase=1)
tableId = p.loadURDF("table/table.urdf", [0.8, 0, 0.0], useFixedBase=1, globalScaling=0.8)
cubeId = p.loadURDF("cube.urdf", [0.65, 0, 0.9], globalScaling=0.08) # Drop it slightly high so it settles

# Let physics settle
for _ in range(100):
    p.stepSimulation()

# --- CONFIGURATION ---
# End Effector Index (Wrist)
ee_index = 16 
# AR10 Hand Joints (From your XML)
hand_indices = []

# Find the hand motor indices again
for i in range(p.getNumJoints(sawyerId)):
    info = p.getJointInfo(sawyerId, i)
    name = info[1].decode("utf-8")
    if "servo" in name:
        hand_indices.append(i)

# Helper: Get Movable Joints for IK
movable_joints = []
for i in range(p.getNumJoints(sawyerId)):
    if p.getJointInfo(sawyerId, i)[2] != p.JOINT_FIXED:
        movable_joints.append(i)

# --- FUNCTIONS ---

def move_arm(target_pos):
    """Calculates IK and moves arm to target_pos"""
    # We force the wrist to point DOWN (approximate) so it doesn't approach sideways
    # This quaternion represents a rotation facing downwards
    target_orn = p.getQuaternionFromEuler([3.14, 0, 0])
    
    joint_poses = p.calculateInverseKinematics(
        sawyerId, 
        ee_index, 
        target_pos, 
        targetOrientation=target_orn
    )
    
    # Apply to motors
    p.setJointMotorControlArray(
        sawyerId, 
        movable_joints, 
        p.POSITION_CONTROL, 
        targetPositions=joint_poses
    )

def control_hand(close_amount):
    """0.0 = Open, 1.5 = Closed"""
    p.setJointMotorControlArray(
        sawyerId, 
        hand_indices, 
        p.POSITION_CONTROL, 
        targetPositions=[close_amount] * len(hand_indices),
        forces=[5000] * len(hand_indices) # Force to hold the cube
    )

# --- MAIN LOOP (The Brain) ---
print("Starting Pick and Place Sequence...")

# Valid offset: Distance from Wrist to Fingertips.
# If the hand crashes into table, INCREASE this. If fingers miss top of cube, DECREASE this.
GRIPPER_LENGTH_OFFSET = 0.25

start_time = time.time()

while True:
    # 1. Get Cube Position (Real-time cheating! We ask the sim where the cube is)
    cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
    cube_x, cube_y, cube_z = cube_pos
    
    # 2. Determine State based on Time
    current_time = time.time() - start_time
    
    if current_time < 3:
        # PHASE 1: HOVER (Go to Cube X,Y but stay high Z)
        target = [cube_x, cube_y, cube_z + 0.3]
        move_arm(target)
        control_hand(0.0) # Open
        
    elif 3 <= current_time < 6:
        # PHASE 2: LOWER (Go down to Cube Z + Offset)
        target = [cube_x, cube_y, cube_z + GRIPPER_LENGTH_OFFSET]
        move_arm(target)
        control_hand(0.0) # Keep Open
        
    elif 6 <= current_time < 8:
        # PHASE 3: GRAB (Stay at bottom, Close Hand)
        target = [cube_x, cube_y, cube_z + GRIPPER_LENGTH_OFFSET]
        move_arm(target)
        control_hand(1.2) # Close firmly
        
    elif current_time >= 8:
        # PHASE 4: LIFT (Go back up to high Z)
        target = [cube_x, cube_y, cube_z + 0.5]
        move_arm(target)
        control_hand(1.2) # Keep Closed
        
    p.stepSimulation()
    time.sleep(1./240.)