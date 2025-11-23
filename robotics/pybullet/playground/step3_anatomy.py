import pybullet as p
import pybullet_data

# 1. Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# 2. Load Sawyer
sawyer_urdf_path = "./sawyer_robot/sawyer_description/urdf/sawyer.urdf"
startPos = [0, 0, 0] 
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

try:
    sawyerId = p.loadURDF(sawyer_urdf_path, startPos, startOrientation, useFixedBase=1)
    print("Sawyer loaded successfully!")
except Exception as e:
    print(f"Error loading Sawyer! Check your path: {sawyer_urdf_path}")
    print(e)

# 3. Anatomy Lesson
# Get the total number of joints
num_joints = p.getNumJoints(sawyerId)
print(f"Total joints detected: {num_joints}")

# Dictionary to map names to IDs
joint_map = {}

print("----------------------------------------------------------------")
print(f"{'ID':<5} {'Joint Name':<30} {'Type (0=Revolute, 4=Fixed)'}")
print("----------------------------------------------------------------")

for i in range(num_joints):
    # getJointInfo returns a long list of data. 
    # Index 1 is the name, Index 2 is the type.
    joint_info = p.getJointInfo(sawyerId, i)
    
    joint_id = joint_info[0]
    joint_name = joint_info[1].decode("utf-8") # It returns bytes, so we decode to string
    joint_type = joint_info[2]
    
    # Filter: We usually only care about joints we can move (Type 0 = Revolute)
    # Fixed joints (Type 4) are just structural glue.
    print(f"{joint_id:<5} {joint_name:<30} {joint_type}")

print("----------------------------------------------------------------")

# Keep window open
while True:
    p.stepSimulation()