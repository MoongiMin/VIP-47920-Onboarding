# ============================================================================
# IMPORTS
# ============================================================================
import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np


# ============================================================================
# PD CONTROLLER
# ============================================================================
# Implements Proportional-Derivative control as specified in README
# Converts desired positions to joint torques
def pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd):
    """
    PD Controller from README specification:
    τ = Kp(X_des - X) + Kd(Ẋ_des - Ẋ)
    
    Args:
        X: Current joint positions
        X_dot: Current joint velocities
        X_des: Desired joint positions
        X_dot_des: Desired joint velocities
        Kp: Proportional gain (controls position error response)
        Kd: Derivative gain (controls velocity error response, damping)
    
    Returns:
        tau: Joint torques to apply
    """
    tau = Kp * (X_des - X) + Kd * (X_dot_des - X_dot)
    return tau


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================
def view_dog():
    """
    Main simulation loop: loads robot model, initializes controller,
    and runs trot gait with PD control
    """
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    # Load MuJoCo model and create data structure
    model = mj.MjModel.from_xml_path("xml/floor2.xml")
    data = mj.MjData(model)

    # Get timestep and launch viewer
    dt = model.opt.timestep
    viewer2 = viewer.launch_passive(model, data)
    
    # Initialize time counter
    t = 0
    
    # ========================================================================
    # PD CONTROLLER PARAMETERS
    # ========================================================================
    # Tuned for smooth locomotion without excessive bouncing
    Kp = 25.0  # Proportional gain - reduced to prevent excessive force/bouncing
    Kd = 6.0   # Derivative gain - high damping to absorb impacts and prevent bouncing
    X_dot_des = np.zeros(12)  # Desired velocities (zero for position control)

    # ========================================================================
    # MAIN CONTROL LOOP
    # ========================================================================
    while viewer2.is_running():
        # Step the simulation forward one timestep
        mj.mj_step(model, data)
        
        # ------------------------------------------------------------------------
        # STATE EXTRACTION
        # ------------------------------------------------------------------------
        # Extract current joint states from simulation
        # qpos contains: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
        # We skip first 7 elements (base position + orientation)
        X = data.qpos[7:]  # Current joint positions (12 joints)
        
        # qvel contains: [vx, vy, vz, ωx, ωy, ωz, joint_vel1, joint_vel2, ...]
        # We skip first 6 elements (linear + angular velocity)
        X_dot = data.qvel[6:]  # Current joint velocities (12 joints)

        # ====================================================================
        # MOVEMENT POLICY: Trot Gait
        # ====================================================================
        # Implements a trot gait where diagonal leg pairs move together:
        # - Diagonal Pair 1: Front-Right + Rear-Left (same phase)
        # - Diagonal Pair 2: Front-Left + Rear-Right (π phase offset)
        # 
        # Each leg cycles through two phases:
        # - SUPPORT PHASE (0 to π): Foot on ground, extends to push body forward
        # - SWING PHASE (π to 2π): Foot in air, flexes and swings forward
        
        # Initialize desired joint position array
        X_des = np.zeros(12)  # 12 joints total: 3 joints × 4 legs
        
        # ------------------------------------------------------------------------
        # GAIT PARAMETERS
        # ------------------------------------------------------------------------
        freq = 3.0  # Gait frequency (Hz) - SLOW to keep feet on ground longer
        
        # Joint angle amplitudes (radians) - increased for better forward/backward oscillation
        hip_lateral = 0.15      # Hip abduction for wide, stable stance (~9°)
        knee_max = 2.0          # Maximum knee flexion during swing (~115°) - INCREASED!
        ankle_flexed = 0.7      # Ankle dorsiflexion during swing (~40°)
        ankle_pushoff = 1.6     # Ankle plantarflexion for push-off (~92°)
        
        # Helper function to compute leg joint angles based on phase
        def compute_leg_angles(phase):
            # Normalize phase to [0, 2π]
            phase_norm = phase % (2 * np.pi)
            
            # SUPPORT PHASE: 0 to π (foot on ground, pulling body forward)
            # Foot position moves: FRONT → BACK (relative to body)
            # This creates forward locomotion as the foot "pulls" the ground backward
            if phase_norm < np.pi:
                # Hip: push outward for wider stance and lateral stability during support
                hip = hip_lateral
                
                # Progress through support phase: 0 (foot front) to 1 (foot back)
                support_progress = phase_norm / np.pi
                
                # Knee: Extended during support, swings from BACK to FRONT
                # Use cosine for smooth transition: starts back, goes forward
                knee = -0.6 * np.cos(support_progress * np.pi)  # -0.6 → +0.6 (back→front)
                
                # Ankle: neutral → gradual plantarflexion for smooth push-off
                # LINEAR progression prevents sudden force spikes that cause bouncing
                ankle = -ankle_pushoff * support_progress  # 0 → -1.6 (linear, smooth)
                
            # SWING PHASE: π to 2π (foot in air, swinging forward for next step)
            # Foot position moves: BACK → FRONT (relative to body)
            # Preparing for next support phase
            else:
                # Adjust phase for swing: 0 to π within swing phase
                swing_phase = phase_norm - np.pi
                swing_progress = swing_phase / np.pi  # 0 to 1
                
                # Hip: maintain outward for stability even during swing
                hip = 0.2  # Keep legs wide for stability (~11 degrees outward)
                
                # Knee: Continues swing FRONT → BACK with flexion for lift
                # Cosine continuation + sin lift for smooth motion
                knee_swing = -0.6 * np.cos((1 + swing_progress) * np.pi)  # +0.6 → -0.6 (front→back)
                knee_lift = 1.2 * np.sin(swing_progress * np.pi)  # 0 → 1.2 → 0 (lift)
                knee = knee_swing + knee_lift  # Combined motion
                
                # Ankle: lifts during swing for ground clearance, sinusoidal
                ankle = -ankle_flexed * np.sin(swing_progress * np.pi)  # 0 → peak → 0
            
            return hip, knee, ankle

        # === Diagonal pair 1: Front-Right (XML's "fl") + Rear-Left (rl) ===
        # Note: XML naming is reversed - "fl" is actually front-right, "fr" is front-left
        phase1 = 2 * np.pi * freq * t  # Calculate phase angle based on time
        _, fr_actual_knee, fr_actual_ankle = compute_leg_angles(phase1)  # Front-Right (XML's fl)
        _, rl_knee, rl_ankle = compute_leg_angles(phase1)  # Rear-Left
        
        # === Diagonal pair 2: Front-Left (XML's "fr") + Rear-Right (rr) ===
        phase2 = phase1 + np.pi  # Add π phase offset (opposite pair)
        _, fl_actual_knee, fl_actual_ankle = compute_leg_angles(phase2)  # Front-Left (XML's fr)
        _, rr_knee, rr_ankle = compute_leg_angles(phase2)  # Rear-Right
        
        # Get base hip values from compute_leg_angles
        base_hip_support = hip_lateral
        base_hip_swing = 0.2
        
        # Determine hip values based on phase (support or swing)
        # Front-Right and Rear-Right: NEGATIVE hip (abduct outward to the right)
        # Front-Left and Rear-Left: POSITIVE hip (abduct outward to the left)
        phase1_support = (phase1 % (2 * np.pi)) < np.pi
        phase2_support = (phase2 % (2 * np.pi)) < np.pi
        
        fr_hip = -(base_hip_support if phase1_support else base_hip_swing)  # Right leg: negative to abduct right
        rr_hip = -(base_hip_support if phase2_support else base_hip_swing)  # Right leg: negative to abduct right
        fl_hip = (base_hip_support if phase2_support else base_hip_swing)   # Left leg: positive to abduct left
        rl_hip = (base_hip_support if phase1_support else base_hip_swing)   # Left leg: positive to abduct left
        
        # Assign desired positions to X_des array [hip, knee, ankle] × 4 legs
        # XML "leg_fl" (indices 0-2) - actually Front-RIGHT
        X_des[0] = fr_hip          # Hip: RIGHT leg abducts to the right (+)
        X_des[1] = fr_actual_knee  # Knee: support vs swing
        X_des[2] = fr_actual_ankle # Ankle: push-off vs clearance
        
        # XML "leg_fr" (indices 3-5) - actually Front-LEFT
        X_des[3] = fl_hip          # Hip: LEFT leg abducts to the left (-)
        X_des[4] = fl_actual_knee  # Knee: support vs swing
        X_des[5] = fl_actual_ankle # Ankle: push-off vs clearance
        
        # XML "leg_rl" (indices 6-8) - Rear-LEFT (correct)
        X_des[6] = rl_hip    # Hip: LEFT leg abducts to the left (-)
        X_des[7] = rl_knee   # Knee: support vs swing
        X_des[8] = rl_ankle  # Ankle: push-off vs clearance
        
        # XML "leg_rr" (indices 9-11) - Rear-RIGHT (correct)
        X_des[9] = rr_hip     # Hip: RIGHT leg abducts to the right (+)
        X_des[10] = rr_knee   # Knee: support vs swing
        X_des[11] = rr_ankle  # Ankle: push-off vs clearance

        # Calculate control torques using PD controller
        tau = pd_controller(X, X_dot, X_des, X_dot_des, Kp, Kd)
        # Apply calculated torques to joint actuators
        data.ctrl[:] = tau
        
        t += dt
        time.sleep(dt)
        viewer2.sync()


if __name__ == "__main__":
    view_dog()
