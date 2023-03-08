import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import time
import math
import torch
np.set_printoptions(suppress=True)


def reset(arm_teacher, arm_student, torque):
    initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    arm_teacher.set_state(initial_state)
    arm_student.set_state(initial_state)

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)

    arm_teacher.set_t(0)
    arm_student.set_t(0)


def set_torque0(arm_teacher, arm_student, torque):
    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)


def score_random_torque(model_path, gui):
    time_limit = 5
    num_tests = 50
    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    arm_teacher = Robot(dynamics_teacher)

    # Student arm
    dynamics_student = ArmDynamicsStudent(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    dynamics_student.init_model(model_path, 3, 0.01, device=torch.device('cpu'))
    arm_student = Robot(dynamics_student)

    scores = []
    torques = np.random.uniform(-1.5, 1.5, num_tests)
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} ( Torque = {torque} Nm)\n')
        reset(arm_teacher, arm_student, torque)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
                # time.sleep(max(0, dt - (time.time() - t)))
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        if gui:
            renderer.plot(None)
            time.sleep(2)

        mse = np.array(mse_list).mean()
        print(f'average mse: {mse}')
        score = 1 if mse < 0.008 else 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50} = {np.array(scores).sum()/50:.2f}')
    print("----------------------------------------\n")


def score_two_torques(model_path, gui):
    time_limit = 5
    num_tests = 50
    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    arm_teacher = Robot(dynamics_teacher)

    # Student arm
    dynamics_student = ArmDynamicsStudent(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    dynamics_student.init_model(model_path, 3, 0.01, device=torch.device('cpu'))
    arm_student = Robot(dynamics_student)

    scores = []
    torques1 = np.random.uniform(-1, 1, num_tests)
    torques2 = np.random.uniform(-1, 1, num_tests)
    for i, (torque1, torque2) in enumerate(zip(torques1, torques2)):
        print("\n----------------------------------------")
        print(f'TEST {i+1} ( Torque 1 = {torque1} Nm,  Torque 1 = {torque2} Nm)\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            if arm_teacher.get_t() < time_limit / 2:
                set_torque0(arm_teacher, arm_student, torque1)
            else:
                set_torque0(arm_teacher, arm_student, torque2)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
                # time.sleep(max(0, dt - (time.time() - t)))
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        if gui:
            renderer.plot(None)
            time.sleep(2)

        mse = np.array(mse_list).mean()
        print(f'average mse: {mse}')
        score = 1 if mse < 0.008 else 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50} = {np.array(scores).sum()/50:.2f}')
    print("----------------------------------------\n")


def score_linear_torques(model_path, gui):
    time_limit = 5
    num_tests = 50
    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    arm_teacher = Robot(dynamics_teacher)

    # Student arm
    dynamics_student = ArmDynamicsStudent(
        num_links=3,
        link_mass=0.1,
        link_length=1,
        joint_viscous_friction=0.1,
        dt=0.01
    )
    dynamics_student.init_model(model_path, 3, 0.01, device=torch.device('cpu'))
    arm_student = Robot(dynamics_student)

    scores = []
    torques = np.random.uniform(0.5, 1.5, num_tests)
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} ( Torque 0 -> {torque} Nm )\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            set_torque0(arm_teacher, arm_student, arm_teacher.get_t() / time_limit * torque)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
                # time.sleep(max(0, dt - (time.time() - t)))
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        if gui:
            renderer.plot(None)
            time.sleep(2)

        mse = np.array(mse_list).mean()
        print(f'average mse: {mse}')
        score = 1 if mse < 0.008 else 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50} = {np.array(scores).sum()/50:.2f}')
    print("----------------------------------------\n")