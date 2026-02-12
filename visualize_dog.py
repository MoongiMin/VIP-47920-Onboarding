import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np


def pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd):
    """
    PD Controller: τ = Kp(X_des - X) + Kd(Ẋ_des - Ẋ)
    """
    position_error = X_des - X
    velocity_error = X_dot_des - X_dot
    tau = Kp * position_error + Kd * velocity_error
    return tau


def view_dog():
    """
    Load floor2.xml (includes dog) and visualize with PD controller and trot gait.
    """
    model = mj.MjModel.from_xml_path("xml/floor2.xml")
    data = mj.MjData(model)

    dt = model.opt.timestep
    viewer2 = viewer.launch_passive(model, data)
    
    t = 0
    
    # PD gains (as recommended in README)
    Kp = 10.0
    Kd = 1.0

    # Desired joint velocities (all zero for position tracking)
    X_dot_des = np.zeros(12)

    while viewer2.is_running():
        mj.mj_step(model, data)
        
        # Get current joint states (skip freejoint: 7 qpos, 6 qvel)
        X = data.qpos[7:]      # Joint positions (12 joints)
        X_dot = data.qvel[6:]  # Joint velocities (12 joints)

        # Trot gait: diagonal legs move together
        # fl+rr (front-left + rear-right) in phase
        # fr+rl (front-right + rear-left) opposite phase
        # Joint order: [fl_hip, fl_knee, fl_ankle, fr_hip, fr_knee, fr_ankle, 
        #               rl_hip, rl_knee, rl_ankle, rr_hip, rr_knee, rr_ankle]
        X_des = np.zeros(12)
        
        freq = 8.0
        
        knee_amp = 2
        ankle_amp = 1.7
        
        # Diagonal pair 1: fl + rr
        phase1 = 2 * np.pi * freq * t
        fl_knee = knee_amp * np.sin(phase1)
        fl_ankle = ankle_amp * np.sin(phase1)
        rr_knee = knee_amp * np.sin(phase1)
        rr_ankle = ankle_amp * np.sin(phase1)
        
        # Diagonal pair 2: fr + rl
        phase2 = phase1 + np.pi * 0.6
        fr_knee = knee_amp * np.sin(phase2)
        fr_ankle = ankle_amp * np.sin(phase2)
        rl_knee = knee_amp * np.sin(phase2)
        rl_ankle = ankle_amp * np.sin(phase2)
        
        X_des[0] = 0
        X_des[1] = fl_knee
        X_des[2] = fl_ankle
        
        X_des[3] = 0
        X_des[4] = fr_knee
        X_des[5] = fr_ankle
        
        X_des[6] = 0
        X_des[7] = rl_knee
        X_des[8] = rl_ankle
        
        X_des[9] = 0
        X_des[10] = rr_knee
        X_des[11] = rr_ankle

        # PD Controller
        tau = pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd)
        
        # Apply control torques
        data.ctrl[:] = tau
        
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    view_dog()
