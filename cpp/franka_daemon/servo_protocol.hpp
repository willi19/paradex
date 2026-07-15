// File: src_main/cpp_sources/daemon/servo_protocol.hpp
// Protocol definition for ThinServoDaemon <-> Python communication
//
// Communication: ZeroMQ + MessagePack
// - REQ/REP socket for commands (port 5555)
// - PUB socket for state streaming (port 5556)
//
// Message format: MessagePack encoded maps with 'type' field for dispatch
//
// Created: 2026-01-24

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace servo_protocol {

// =============================================================================
// MESSAGE TYPES
// =============================================================================

// Command types (REQ/REP)
// -- Motion commands --
constexpr const char* CMD_MOVE_TO_QPOS = "move_to_qpos";
constexpr const char* CMD_MOVE_TO_CARTESIAN = "move_to_cartesian";
constexpr const char* CMD_SET_CARTESIAN_VELOCITY = "set_cartesian_velocity";
constexpr const char* CMD_SET_JOINT_VELOCITY = "set_joint_velocity";
// -- Torque / impedance control --
constexpr const char* CMD_SET_TORQUES = "set_torques";
constexpr const char* CMD_SET_JOINT_IMPEDANCE = "set_joint_impedance";
constexpr const char* CMD_SET_CARTESIAN_IMPEDANCE = "set_cartesian_impedance";
constexpr const char* CMD_SET_COLLISION_BEHAVIOR = "set_collision_behavior";
// -- Robot configuration --
constexpr const char* CMD_SET_LOAD = "set_load";
constexpr const char* CMD_SET_EE = "set_ee";
constexpr const char* CMD_SET_GUIDING_MODE = "set_guiding_mode";
constexpr const char* CMD_ERROR_RECOVERY = "error_recovery";
// -- Gripper --
constexpr const char* CMD_OPEN_GRIPPER = "open_gripper";
constexpr const char* CMD_GRASP = "grasp";
// -- Streaming velocity/torque control --
constexpr const char* CMD_STOP_STREAMING = "stop_streaming";
// -- State / control --
constexpr const char* CMD_GET_STATE = "get_state";
constexpr const char* CMD_STOP = "stop";
constexpr const char* CMD_PING = "ping";
constexpr const char* CMD_SHUTDOWN = "shutdown";

// Response types
constexpr const char* RSP_SUCCESS = "success";
constexpr const char* RSP_ERROR = "error";

// State stream type (PUB)
constexpr const char* STATE_UPDATE = "state_update";

// =============================================================================
// DEFAULT PORTS
// =============================================================================

constexpr int DEFAULT_COMMAND_PORT = 5555;
constexpr int DEFAULT_STATE_PORT = 5556;

// =============================================================================
// STRUCTURES (for documentation - actual serialization uses MessagePack)
// =============================================================================

// MoveToQpos Request
// Fields:
//   type: "move_to_qpos"
//   qpos: array[7] - target joint positions (rad)
//   speed_scale: double - joint speed scale (0.0-1.0)
//   collision_torque: double - collision threshold (Nm)
//   collision_force: double - collision threshold (N)

// MoveToCartesian Request
// Fields:
//   type: "move_to_cartesian"
//   position: array[3] - target position (m) in base frame
//   orientation: array[9] - rotation matrix (row-major) in base frame
//   move_speed: double - Cartesian speed (m/s)
//   collision_torque: double
//   collision_force: double

// SetCartesianVelocity Request
// Fields:
//   type: "set_cartesian_velocity"
//   twist: array[6] - [vx, vy, vz, wx, wy, wz] in base frame
//   duration_ms: int - how long to apply velocity (0 = until next command)

// OpenGripper Request
// Fields:
//   type: "open_gripper"
//   width: double - target width (m)
//   speed: double - opening speed (m/s)

// Grasp Request
// Fields:
//   type: "grasp"
//   force: double - grasp force (N)
//   speed: double - closing speed (m/s)
//   eps_inner: double - inner tolerance (m)
//   eps_outer: double - outer tolerance (m)

// SetJointVelocity Request
// Fields:
//   type: "set_joint_velocity"
//   dq: array[7] - joint velocities (rad/s)
//   duration_ms: int - how long to apply (ms)

// SetTorques Request
// Fields:
//   type: "set_torques"
//   torques: array[7] - desired joint torques (Nm)
//   duration_ms: int - how long to apply (ms)

// SetJointImpedance Request
// Fields:
//   type: "set_joint_impedance"
//   K_theta: array[7] - joint stiffness (Nm/rad)

// SetCartesianImpedance Request
// Fields:
//   type: "set_cartesian_impedance"
//   K_x: array[6] - Cartesian stiffness [x,y,z,roll,pitch,yaw]

// SetCollisionBehavior Request
// Fields:
//   type: "set_collision_behavior"
//   torque_lower: array[7] - lower torque thresholds (Nm)
//   torque_upper: array[7] - upper torque thresholds (Nm)
//   force_lower: array[6] - lower force thresholds (N/Nm)
//   force_upper: array[6] - upper force thresholds (N/Nm)

// SetLoad Request
// Fields:
//   type: "set_load"
//   mass: double - payload mass (kg)
//   F_x_Cload: array[3] - center of mass in flange frame (m)
//   load_inertia: array[9] - inertia matrix (row-major, kg*m^2)

// SetEE Request
// Fields:
//   type: "set_ee"
//   NE_T_EE: array[16] - EE transformation (4x4 column-major)

// SetGuidingMode Request
// Fields:
//   type: "set_guiding_mode"
//   guiding_axes: array[6] - [x,y,z,roll,pitch,yaw] true=free
//   nullspace: bool - allow nullspace motion

// ErrorRecovery Request
// Fields:
//   type: "error_recovery"

// GetState Request
// Fields:
//   type: "get_state"

// Success Response
// Fields:
//   type: "success"
//   ... additional fields depending on command

// Error Response
// Fields:
//   type: "error"
//   message: string - error description

// State Update (streamed via PUB)
// Fields:
//   type: "state_update"
//   timestamp: double - unix timestamp (s)
//   qpos: array[7] - joint positions (rad)
//   qvel: array[7] - joint velocities (rad/s)
//   tau_ext: array[7] - external torques (Nm)
//   wrench: array[6] - F/T sensor [Fx, Fy, Fz, Tx, Ty, Tz] in EE frame
//   O_T_EE: array[16] - EE pose (4x4 column-major) in base frame
//   gripper_width: double - current gripper width (m)
//   gripper_grasping: bool - true if grasping object

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Convert between ZeroMQ endpoint strings
inline std::string tcp_endpoint(int port) {
    return "tcp://*:" + std::to_string(port);
}

inline std::string tcp_connect(const std::string& host, int port) {
    return "tcp://" + host + ":" + std::to_string(port);
}

}  // namespace servo_protocol
