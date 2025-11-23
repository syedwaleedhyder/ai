import pybullet as p
import pybullet_data
import time

# 1. Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# 2. Load Sawyer
startPos = [0, 0, 0] 
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
sawyer_urdf_path = "./sawyer_robot/sawyer_description/urdf/sawyer.urdf"
try:
    sawyerId = p.loadURDF(sawyer_urdf_path, startPos, startOrientation, useFixedBase=1)
    print("Sawyer loaded successfully!")
except Exception as e:
    print(f"Error loading Sawyer! Check your path: {sawyer_urdf_path}")
    print(e)

# 3. Create Sliders for Every Movable Joint
# We store the slider ID and the joint ID together so we can link them later.
active_joints = [] 

num_joints = p.getNumJoints(sawyerId)

for i in range(num_joints):
    joint_info = p.getJointInfo(sawyerId, i)
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]
    
    # We only want to control REVOLUTE joints (Type 0)
    if joint_type == p.JOINT_REVOLUTE:
        # arguments: user_text, range_min, range_max, start_value
        # We set the range from -3.14 to +3.14 (approx -180 to +180 degrees)
        slider_id = p.addUserDebugParameter(joint_name, -3.14, 3.14, 0)
        
        # Save the mapping: [joint_index, slider_id]
        active_joints.append((i, slider_id))

print(f"Created sliders for {len(active_joints)} joints.")

# 4. The Control Loop
while True:
    # For every active joint...
    for joint_index, slider_id in active_joints:
        # a. Read the value from the GUI slider
        target_angle = p.readUserDebugParameter(slider_id)
        
        # b. Send the command to the motor
        p.setJointMotorControl2(
            bodyUniqueId=sawyerId,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle
        )
        
    p.stepSimulation()
    time.sleep(1./240.)