# PyBullet Robotics Tutorial - Sawyer Robot

A step-by-step tutorial for learning robotics simulation with PyBullet using the Sawyer robot. This project covers everything from basic physics simulation to advanced pick-and-place operations.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Tutorial Steps](#tutorial-steps)

## 🎯 Overview

This tutorial provides hands-on experience with:
- PyBullet physics simulation
- Robot loading and visualization
- Forward kinematics (FK)
- Inverse kinematics (IK)
- Robot hand control
- Environment setup
- Pick-and-place operations

## 📦 Requirements

- Python 3.7+
- PyBullet (`pip install pybullet`)
- NumPy (usually comes with PyBullet)

## 📁 Project Structure

```

├── step1_basics.py     # Basic PyBullet setup
├── step2_sawyer.py     # Loading Sawyer robot
├── step3_anatomy.py    # Understanding robot joints
├── step4_sliders_fk.py # Forward kinematics with sliders
├── step5_ik.py         # Inverse kinematics
├── step6_ar10_hand.py  # AR10 hand control
├── step7_environment.py # Environment setup
└── step8_pick_object.py # Pick and place task
```

## 📚 Tutorial Steps

### Step 1: Basics
**[`step1_basics.py`](step1_basics.py)**

Introduction to PyBullet:
- Connecting to the physics server
- Loading basic objects (plane, cube)
- Setting up gravity
- Running a simulation loop

**Key Concepts:**
- `p.connect(p.GUI)` - Start graphical interface
- `p.loadURDF()` - Load robot/object files
- `p.stepSimulation()` - Advance physics simulation

---

### Step 2: Loading Sawyer
**[`step2_sawyer.py`](step2_sawyer.py)**

Load the Sawyer robot into the simulation:
- Setting up the environment
- Loading the Sawyer URDF file
- Positioning the robot
- Error handling for file paths

**Key Concepts:**
- `useFixedBase=1` - Fix robot base to world
- URDF file paths and loading
- Robot positioning and orientation

---

### Step 3: Robot Anatomy
**[`step3_anatomy.py`](step3_anatomy.py)**

Understanding the robot's structure:
- Enumerating all joints
- Identifying joint types (Revolute vs Fixed)
- Mapping joint names to IDs
- Understanding the robot's kinematic chain

**Key Concepts:**
- `p.getNumJoints()` - Get total joint count
- `p.getJointInfo()` - Get joint information
- Joint types: `JOINT_REVOLUTE` (0) vs `JOINT_FIXED` (4)

---

### Step 4: Forward Kinematics with Sliders
**[`step4_sliders_fk.py`](step4_sliders_fk.py)**

Interactive robot control:
- Creating GUI sliders for each joint
- Controlling joint angles in real-time
- Understanding forward kinematics
- Visual feedback of robot movement

**Key Concepts:**
- `p.addUserDebugParameter()` - Create GUI sliders
- `p.readUserDebugParameter()` - Read slider values
- `p.setJointMotorControl2()` - Control joint positions
- Position control mode

---

### Step 5: Inverse Kinematics
**[`step5_ik.py`](step5_ik.py)**

Move the robot to target positions:
- Setting target end-effector positions
- Using PyBullet's IK solver
- Mapping IK solutions to joint motors
- Visual debugging with debug lines

**Key Concepts:**
- `p.calculateInverseKinematics()` - Solve IK problem
- End-effector control
- Target position specification
- Joint angle mapping

---

### Step 6: AR10 Hand Control
**[`step6_ar10_hand.py`](step6_ar10_hand.py)**

Controlling the robot's hand:
- Identifying hand joint indices
- Coordinated finger control
- Opening and closing the gripper
- Arm positioning for hand visibility

**Key Concepts:**
- Hand joint identification
- Coordinated multi-joint control
- `p.setJointMotorControlArray()` - Control multiple joints
- Force/torque limits

---

### Step 7: Environment Setup
**[`step7_environment.py`](step7_environment.py)**

Creating a complete simulation environment:
- Loading tables and objects
- Positioning multiple objects
- Scaling objects appropriately
- Camera positioning
- Physics settling

**Key Concepts:**
- Multi-object environments
- Object scaling (`globalScaling`)
- Camera control
- Physics initialization

---

### Step 8: Pick and Place
**[`step8_pick_object.py`](step8_pick_object.py)**

Complete pick-and-place task:
- Combining IK and hand control
- State machine for task execution
- Real-time object tracking
- Coordinated arm and hand movements

**Key Concepts:**
- Task state machines
- `p.getBasePositionAndOrientation()` - Get object pose
- Coordinated arm and gripper control
- End-effector orientation control

---
