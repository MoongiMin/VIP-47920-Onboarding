import time
import mujoco as mj
from mujoco import viewer
import numpy as np


def pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd):
    tau = Kp * (X_des - X) + Kd * (X_dot_des - X_dot)
    return tau


def view_dog():
    model = mj.MjModel.from_xml_path("xml/floor2.xml")
    data = mj.MjData(model)

    dt = model.opt.timestep
    viewer2 = viewer.launch_passive(model, data)
    
    t = 0
    
    Kp = 25.0  # Proportional gain
    Kd = 6.0   # Derivative gain
    X_dot_des = np.zeros(12)  # Desired velocities (zero for position control)

    while True:
        mj.mj_step(model, data)
        
        X = data.qpos[7:]  # Current joint positions (12 joints)
        X_dot = data.qvel[6:]  # Current joint velocities (12 joints)

        X_des = np.zeros(12)  # 12 joints total: 3 joints × 4 legs
        
        freq = 3.0  # Gait frequency (Hz)
        
        hip_lateral = 0.15      # Hip abduction for wide, stable stance
        knee_max = 2.0          # Maximum knee flexion during swing 
        ankle_flexed = 0.7      # Ankle dorsiflexion during swing 
        ankle_pushoff = 1.6     # Ankle plantarflexion for push-off 
        def compute_leg_angles(phase):
            phase_norm = phase % (2 * np.pi)
            
            if phase_norm < np.pi:
                hip = hip_lateral
                support_progress = phase_norm / np.pi
                knee = -0.6 * np.cos(support_progress * np.pi)  # -0.6 → +0.6 (back→front)
                ankle = -ankle_pushoff * support_progress  # 0 → -1.6 (linear, smooth)
            else:
                swing_phase = phase_norm - np.pi
                swing_progress = swing_phase / np.pi  # 0 to 1
                hip = 0.2  # Keep legs wide for stability (~11 degrees outward)
                knee_swing = -0.6 * np.cos((1 + swing_progress) * np.pi)  # +0.6 → -0.6 (front→back)
                knee_lift = 1.2 * np.sin(swing_progress * np.pi)  # 0 → 1.2 → 0 (lift)
                knee = knee_swing + knee_lift  # Combined motion
                ankle = -ankle_flexed * np.sin(swing_progress * np.pi)  # 0 → peak → 0
            
            return hip, knee, ankle

        phase1 = 2 * np.pi * freq * t  # Calculate phase angle based on time
        _, fr_actual_knee, fr_actual_ankle = compute_leg_angles(phase1)  # Front-Right (XML's fl)
        _, rl_knee, rl_ankle = compute_leg_angles(phase1)  # Rear-Left
        
        phase2 = phase1 + np.pi  # Add π phase offset (opposite pair)
        _, fl_actual_knee, fl_actual_ankle = compute_leg_angles(phase2)  # Front-Left (XML's fr)
        _, rr_knee, rr_ankle = compute_leg_angles(phase2)  # Rear-Right
        
        base_hip_support = hip_lateral
        base_hip_swing = 0.2
        
        phase1_support = (phase1 % (2 * np.pi)) < np.pi
        phase2_support = (phase2 % (2 * np.pi)) < np.pi
        
        fr_hip = -(base_hip_support if phase1_support else base_hip_swing)  # Right leg: negative to abduct right
        rr_hip = -(base_hip_support if phase2_support else base_hip_swing)  # Right leg: negative to abduct right
        fl_hip = (base_hip_support if phase2_support else base_hip_swing)   # Left leg: positive to abduct left
        rl_hip = (base_hip_support if phase1_support else base_hip_swing)   # Left leg: positive to abduct left
        
        # XML "leg_fl" (indices 0-2) - actually Front-RIGHT
        X_des[0] = fr_hip          # Hip: RIGHT leg abducts to the right (+)
        X_des[1] = fr_actual_knee  # Knee: support vs swing
        X_des[2] = fr_actual_ankle # Ankle: push-off vs clearance
        
        # XML "leg_fr" (indices 3-5) - actually Front-LEFT
        X_des[3] = fl_hip          # Hip: LEFT leg abducts to the left (-)
        X_des[4] = fl_actual_knee  # Knee: support vs swing
        X_des[5] = fl_actual_ankle # Ankle: push-off vs clearance
        
        # XML "leg_rl" (indices 6-8) - Rear-LEFT 
        X_des[6] = rl_hip    # Hip: LEFT leg abducts to the left (-)
        X_des[7] = rl_knee   # Knee: support vs swing
        X_des[8] = rl_ankle  # Ankle: push-off vs clearance
        
        # XML "leg_rr" (indices 9-11) - Rear-RIGHT 
        X_des[9] = rr_hip     # Hip: RIGHT leg abducts to the right (+)
        X_des[10] = rr_knee   # Knee: support vs swing
        X_des[11] = rr_ankle  # Ankle: push-off vs clearance

        tau = pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd)
        data.ctrl[:] = tau
        
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    view_dog()
