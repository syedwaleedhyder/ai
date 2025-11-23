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


# 3. Define the "End Effector"
# This is the part of the robot we want to control. 
# Based on your previous check, Index 16 is the wrist (right_j6).
end_effector_index = 16 

# 4. Create Target Sliders (X, Y, Z)
# This defines where in space we want the hand to go.
# Start slightly in front of the robot (x=0.5) and up (z=0.5)
slider_x = p.addUserDebugParameter("Target X", -1.0, 1.0, 0.5)
slider_y = p.addUserDebugParameter("Target Y", -1.0, 1.0, 0.0)
slider_z = p.addUserDebugParameter("Target Z", 0.0, 1.5, 0.5)

print("Move the sliders to tell the robot where to go!")

while True:
    # a. Read the sliders
    x = p.readUserDebugParameter(slider_x)
    y = p.readUserDebugParameter(slider_y)
    z = p.readUserDebugParameter(slider_z)
    target_pos = [x, y, z]

    # b. Visual Debugging
    # Draw a red line from the target to slightly above the target
    # This puts a visual marker in the world so you know where you are aiming.
    p.addUserDebugLine(target_pos, [x, y, z+0.1], [1, 0, 0], lifeTime=0.1)

    # c. Calculate Inverse Kinematics
    # This is the magic function.
    joint_angles = p.calculateInverseKinematics(
        sawyerId, 
        end_effector_index, 
        target_pos
    )

    # d. Apply the calculated angles to the motors
    # calculateInverseKinematics returns a list of angles for ALL joints.
    # Note: The sawyer URDF has some fixed joints, so mapping the output 
    # to the motors requires matching the movable joints.
    
    # Get all movable joints
    movable_joints = []
    for i in range(p.getNumJoints(sawyerId)):
        if p.getJointInfo(sawyerId, i)[2] != p.JOINT_FIXED:
            movable_joints.append(i)

    # Apply the angles
    # The length of joint_angles matches the number of movable joints
    p.setJointMotorControlArray(
        sawyerId, 
        movable_joints, 
        p.POSITION_CONTROL, 
        targetPositions=joint_angles
    )

    p.stepSimulation()