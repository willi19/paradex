// File: src_main/cpp_sources/daemon/franka_daemon.cpp
// FrankaDaemon - C++ daemon bridging Python ↔ libfranka via ZMQ + msgpack
//
// Supports: motion (joint/cartesian position & velocity), torque control,
//           impedance configuration, gripper, guiding mode, EE/load config,
//           collision behavior, error recovery, 100Hz state streaming.
//
// All FSM logic stays in Python. This daemon executes primitives.
//
// Communication: ZeroMQ + MessagePack
// - REQ/REP socket (port 5555) for commands
// - PUB socket (port 5556) for state streaming
//
// Usage: ./franka_daemon <robot-ip> [--command_port PORT] [--state_port PORT]
//
// Created: 2026-01-24

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>

#include <zmq.hpp>
#include <msgpack.hpp>

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../modules/franka_servo.hpp"
#include "../modules/robot_params.hpp"
#include "servo_protocol.hpp"

// Configuration path for unified Python-C++ parameters
static const std::string CONFIG_PATH = "/tmp/robot_params.json";

// =============================================================================
// GLOBALS
// =============================================================================

static std::atomic<bool> g_running{true};
static std::mutex g_robot_mutex;

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    std::cout << "\n[DAEMON] Received signal " << signum << ", shutting down...\n";
    g_running = false;
}

// =============================================================================
// STATE STRUCTURE
// =============================================================================

struct RobotStateSnapshot {
    double timestamp;
    std::array<double, 7> qpos;
    std::array<double, 7> qvel;
    std::array<double, 7> tau_ext;
    std::array<double, 6> wrench;  // F/T in EE frame
    std::array<double, 16> O_T_EE;  // EE pose (4x4 column-major)
    double gripper_width;
    bool gripper_grasping;
};

// =============================================================================
// MSGPACK SERIALIZATION HELPERS
// =============================================================================

// Serialize response to msgpack
msgpack::sbuffer pack_success_response(const std::string& extra_info = "") {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(2);
    pk.pack("type");
    pk.pack(servo_protocol::RSP_SUCCESS);
    pk.pack("info");
    pk.pack(extra_info);
    return buffer;
}

msgpack::sbuffer pack_error_response(const std::string& message) {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(2);
    pk.pack("type");
    pk.pack(servo_protocol::RSP_ERROR);
    pk.pack("message");
    pk.pack(message);
    return buffer;
}

msgpack::sbuffer pack_state_response(const RobotStateSnapshot& state) {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(9);

    pk.pack("type");
    pk.pack(servo_protocol::STATE_UPDATE);

    pk.pack("timestamp");
    pk.pack(state.timestamp);

    pk.pack("qpos");
    pk.pack_array(7);
    for (int i = 0; i < 7; ++i) pk.pack(state.qpos[i]);

    pk.pack("qvel");
    pk.pack_array(7);
    for (int i = 0; i < 7; ++i) pk.pack(state.qvel[i]);

    pk.pack("tau_ext");
    pk.pack_array(7);
    for (int i = 0; i < 7; ++i) pk.pack(state.tau_ext[i]);

    pk.pack("wrench");
    pk.pack_array(6);
    for (int i = 0; i < 6; ++i) pk.pack(state.wrench[i]);

    pk.pack("O_T_EE");
    pk.pack_array(16);
    for (int i = 0; i < 16; ++i) pk.pack(state.O_T_EE[i]);

    pk.pack("gripper_width");
    pk.pack(state.gripper_width);

    pk.pack("gripper_grasping");
    pk.pack(state.gripper_grasping);

    return buffer;
}

msgpack::sbuffer pack_grasp_response(bool success, bool object_detected, double final_width) {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(4);
    pk.pack("type");
    pk.pack(servo_protocol::RSP_SUCCESS);
    pk.pack("object_detected");
    pk.pack(object_detected);
    pk.pack("final_width");
    pk.pack(final_width);
    pk.pack("success");
    pk.pack(success);
    return buffer;
}

// =============================================================================
// COMMAND HANDLERS
// =============================================================================

class CommandHandler {
public:
    CommandHandler(franka::Robot& robot, franka::Gripper* gripper)
        : robot_(robot), gripper_(gripper),
          last_gripper_width_(0.08), is_grasping_(false),
          connection_healthy_(true),
          streaming_active_(false),
          stream_mode_(StreamMode::NONE),
          stream_state_valid_(false) {
        stream_dq_target_.fill(0.0);
        stream_twist_target_.fill(0.0);
        stream_torque_target_.fill(0.0);
    }

    ~CommandHandler() {
        stop_streaming_internal();
    }

    enum class StreamMode { NONE, JOINT_VEL, CART_VEL, TORQUE };

    msgpack::sbuffer handle_command(const msgpack::object& obj) {
        try {
            if (obj.type != msgpack::type::MAP) {
                return pack_error_response("Invalid message format: expected map");
            }

            std::map<std::string, msgpack::object> msg;
            obj.convert(msg);

            auto it = msg.find("type");
            if (it == msg.end()) {
                return pack_error_response("Missing 'type' field");
            }

            std::string cmd_type = it->second.as<std::string>();

            if (cmd_type == servo_protocol::CMD_PING) {
                return handle_ping();
            } else if (cmd_type == servo_protocol::CMD_GET_STATE) {
                return handle_get_state();
            } else if (cmd_type == servo_protocol::CMD_MOVE_TO_QPOS) {
                return handle_move_to_qpos(msg);
            } else if (cmd_type == servo_protocol::CMD_MOVE_TO_CARTESIAN) {
                return handle_move_to_cartesian(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_CARTESIAN_VELOCITY) {
                return handle_set_cartesian_velocity(msg);
            } else if (cmd_type == servo_protocol::CMD_STOP_STREAMING) {
                return handle_stop_streaming();
            } else if (cmd_type == servo_protocol::CMD_OPEN_GRIPPER) {
                return handle_open_gripper(msg);
            } else if (cmd_type == servo_protocol::CMD_GRASP) {
                return handle_grasp(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_JOINT_VELOCITY) {
                return handle_set_joint_velocity(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_TORQUES) {
                return handle_set_torques(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_JOINT_IMPEDANCE) {
                return handle_set_joint_impedance(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_CARTESIAN_IMPEDANCE) {
                return handle_set_cartesian_impedance(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_COLLISION_BEHAVIOR) {
                return handle_set_collision_behavior(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_LOAD) {
                return handle_set_load(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_EE) {
                return handle_set_ee(msg);
            } else if (cmd_type == servo_protocol::CMD_SET_GUIDING_MODE) {
                return handle_set_guiding_mode(msg);
            } else if (cmd_type == servo_protocol::CMD_ERROR_RECOVERY) {
                return handle_error_recovery();
            } else if (cmd_type == servo_protocol::CMD_STOP) {
                return handle_stop();
            } else if (cmd_type == servo_protocol::CMD_SHUTDOWN) {
                g_running = false;
                return pack_success_response("Shutdown initiated");
            } else {
                return pack_error_response("Unknown command type: " + cmd_type);
            }
        } catch (const std::exception& e) {
            return pack_error_response(std::string("Exception: ") + e.what());
        }
    }

    RobotStateSnapshot get_current_state() {
        // During streaming control, use state from control callback (1kHz)
        // instead of readOnce() which would conflict with the active control loop
        if (stream_state_valid_.load()) {
            std::lock_guard<std::mutex> lock(stream_state_mutex_);
            return stream_state_;
        }

        // If streaming is active or a stream mode is set (thread starting/stopping),
        // NEVER call readOnce() — it WILL conflict with robot_.control().
        // Wait for stream state, or return empty state as last resort.
        if (streaming_active_.load() || stream_mode_.load() != StreamMode::NONE) {
            for (int i = 0; i < 500; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (stream_state_valid_.load()) {
                    std::lock_guard<std::mutex> lock(stream_state_mutex_);
                    return stream_state_;
                }
                // Only break if streaming is fully done
                if (!streaming_active_.load() && stream_mode_.load() == StreamMode::NONE) break;
            }
            // Stream died without producing state — return empty (do NOT readOnce)
            if (streaming_active_.load() || stream_mode_.load() != StreamMode::NONE) {
                RobotStateSnapshot state{};
                state.timestamp = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                state.gripper_width = last_gripper_width_;
                state.gripper_grasping = is_grasping_;
                return state;
            }
        }

        // No streaming active — safe to call readOnce()
        std::lock_guard<std::mutex> lock(g_robot_mutex);

        RobotStateSnapshot state;
        state.timestamp = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        try {
            franka::RobotState rs = robot_.readOnce();
            for (int i = 0; i < 7; ++i) {
                state.qpos[i] = rs.q[i];
                state.qvel[i] = rs.dq[i];
                state.tau_ext[i] = rs.tau_ext_hat_filtered[i];
            }
            for (int i = 0; i < 6; ++i) {
                state.wrench[i] = rs.O_F_ext_hat_K[i];
            }
            for (int i = 0; i < 16; ++i) {
                state.O_T_EE[i] = rs.O_T_EE[i];
            }
        } catch (const franka::Exception& e) {
            std::cerr << "[STATE] Failed to read robot state: " << e.what() << "\n";
            state.qpos.fill(0);
            state.qvel.fill(0);
            state.tau_ext.fill(0);
            state.wrench.fill(0);
            state.O_T_EE.fill(0);
        }

        state.gripper_width = last_gripper_width_;
        state.gripper_grasping = is_grasping_;

        return state;
    }

private:
    franka::Robot& robot_;
    franka::Gripper* gripper_;  // nullable: gripper may not be connected
    double last_gripper_width_;
    bool is_grasping_;
    bool connection_healthy_;  // Track if robot connection is healthy

    // Streaming control state
    std::atomic<bool> streaming_active_;
    std::thread streaming_thread_;
    std::mutex stream_mutex_;
    std::array<double, 7> stream_dq_target_;
    std::array<double, 6> stream_twist_target_;
    std::array<double, 7> stream_torque_target_;
    std::atomic<StreamMode> stream_mode_;

    // Latest state from streaming control callback (updated at 1kHz)
    std::mutex stream_state_mutex_;
    RobotStateSnapshot stream_state_;
    std::atomic<bool> stream_state_valid_;

    // ----------------------------------------------------------------
    // Streaming control thread functions
    // ----------------------------------------------------------------

    void stop_streaming_internal() {
        streaming_active_ = false;  // Signal thread to ramp down and exit
        if (streaming_thread_.joinable()) {
            streaming_thread_.join();
        }
        stream_mode_ = StreamMode::NONE;
        stream_state_valid_ = false;
    }

    void update_stream_state_from_robot(const franka::RobotState& rs) {
        std::lock_guard<std::mutex> lock(stream_state_mutex_);
        stream_state_.timestamp = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        for (int i = 0; i < 7; ++i) {
            stream_state_.qpos[i] = rs.q[i];
            stream_state_.qvel[i] = rs.dq[i];
            stream_state_.tau_ext[i] = rs.tau_ext_hat_filtered[i];
        }
        for (int i = 0; i < 6; ++i) {
            stream_state_.wrench[i] = rs.O_F_ext_hat_K[i];
        }
        for (int i = 0; i < 16; ++i) {
            stream_state_.O_T_EE[i] = rs.O_T_EE[i];
        }
        stream_state_.gripper_width = last_gripper_width_;
        stream_state_.gripper_grasping = is_grasping_;
        stream_state_valid_ = true;
    }

    void run_joint_velocity_stream() {
        std::array<double, 7> current_dq{};  // starts at 0 → smooth ramp-up
        const double max_ddq = 10.0;  // rad/s² rate limit per joint

        // Hold g_robot_mutex for entire robot_.control() — prevents state thread's
        // readOnce() from conflicting. State thread uses stream_state_ during streaming.
        {
        std::lock_guard<std::mutex> lock(g_robot_mutex);
        std::cout << "[STREAM] Joint velocity streaming started" << std::endl;
        try {
            robot_.control([&](const franka::RobotState& rs,
                              franka::Duration period) -> franka::JointVelocities {
                double dt = period.toSec();
                if (dt <= 0) dt = 0.001;

                update_stream_state_from_robot(rs);

                // Get target
                std::array<double, 7> target;
                {
                    std::lock_guard<std::mutex> lock(stream_mutex_);
                    target = stream_dq_target_;
                }

                if (!streaming_active_.load()) {
                    target.fill(0.0);  // ramp to zero when stopping
                }

                // Rate-limited transition (smooth acceleration)
                for (int i = 0; i < 7; ++i) {
                    double diff = target[i] - current_dq[i];
                    double max_step = max_ddq * dt;
                    if (diff > max_step) diff = max_step;
                    if (diff < -max_step) diff = -max_step;
                    current_dq[i] += diff;
                }

                // Finish when stopped and velocity near zero
                if (!streaming_active_.load()) {
                    double sum = 0;
                    for (int i = 0; i < 7; ++i) sum += std::abs(current_dq[i]);
                    if (sum < 1e-5) {
                        std::cout << "[STREAM] Joint velocity stream finished (ramped to zero)\n";
                        return franka::MotionFinished(
                            franka::JointVelocities({0, 0, 0, 0, 0, 0, 0}));
                    }
                }

                return franka::JointVelocities({
                    current_dq[0], current_dq[1], current_dq[2], current_dq[3],
                    current_dq[4], current_dq[5], current_dq[6]});
            });
        } catch (const franka::Exception& e) {
            std::cout << "[STREAM] Joint velocity error: " << e.what() << std::endl;
            connection_healthy_ = false;
            try { robot_.automaticErrorRecovery(); connection_healthy_ = true; }
            catch (...) {}
        }
        } // g_robot_mutex released
        streaming_active_ = false;
        stream_state_valid_ = false;
        stream_mode_ = StreamMode::NONE;
        std::cout << "[STREAM] Joint velocity thread exiting" << std::endl;
    }

    void run_cartesian_velocity_stream() {
        std::array<double, 6> current_twist{};  // starts at 0
        const double max_acc_lin = 5.0;   // m/s² linear rate limit
        const double max_acc_ang = 10.0;  // rad/s² angular rate limit

        // Hold g_robot_mutex for entire robot_.control() — prevents state thread's
        // readOnce() from conflicting. State thread uses stream_state_ during streaming.
        {
        std::lock_guard<std::mutex> lock(g_robot_mutex);
        std::cout << "[STREAM] Cartesian velocity streaming started" << std::endl;
        try {
            robot_.control([&](const franka::RobotState& rs,
                              franka::Duration period) -> franka::CartesianVelocities {
                double dt = period.toSec();
                if (dt <= 0) dt = 0.001;

                update_stream_state_from_robot(rs);

                std::array<double, 6> target;
                {
                    std::lock_guard<std::mutex> lock(stream_mutex_);
                    target = stream_twist_target_;
                }

                if (!streaming_active_.load()) {
                    target.fill(0.0);
                }

                // Rate-limited transition
                for (int i = 0; i < 6; ++i) {
                    double max_step = (i < 3 ? max_acc_lin : max_acc_ang) * dt;
                    double diff = target[i] - current_twist[i];
                    if (diff > max_step) diff = max_step;
                    if (diff < -max_step) diff = -max_step;
                    current_twist[i] += diff;
                }

                if (!streaming_active_.load()) {
                    double sum = 0;
                    for (int i = 0; i < 6; ++i) sum += std::abs(current_twist[i]);
                    if (sum < 1e-5) {
                        std::cout << "[STREAM] Cartesian velocity stream finished\n";
                        return franka::MotionFinished(
                            franka::CartesianVelocities({0, 0, 0, 0, 0, 0}));
                    }
                }

                return franka::CartesianVelocities({
                    current_twist[0], current_twist[1], current_twist[2],
                    current_twist[3], current_twist[4], current_twist[5]});
            });
        } catch (const franka::Exception& e) {
            std::cout << "[STREAM] Cartesian velocity error: " << e.what() << std::endl;
            connection_healthy_ = false;
            try { robot_.automaticErrorRecovery(); connection_healthy_ = true; }
            catch (...) {}
        }
        } // g_robot_mutex released
        streaming_active_ = false;
        stream_state_valid_ = false;
        stream_mode_ = StreamMode::NONE;
        std::cout << "[STREAM] Cartesian velocity thread exiting" << std::endl;
    }

    void run_torque_stream() {
        std::array<double, 7> current_tau{};  // starts at 0
        const double max_dtau = 500.0;  // Nm/s torque rate limit
        int callback_count = 0;

        // FR3 joint velocity limits (rad/s) — firmware triggers reflex beyond these
        static constexpr double DQ_MAX[7] = {2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610};
        const double dq_safety = 0.6;   // start damping at 60% of firmware limit
        const double Kd_vel   = 40.0;   // velocity damping gain (Nm·s/rad)

        // Hold g_robot_mutex for entire robot_.control() — prevents state thread's
        // readOnce() from conflicting. State thread uses stream_state_ during streaming.
        {
        std::lock_guard<std::mutex> lock(g_robot_mutex);

        // Use COMBINED motion+torque interface to avoid joint_velocity_violation.
        //
        // Pure torque mode (robot_.control(torque_cb)) turns OFF the internal
        // impedance controller.  During the mode transition, joints under gravity
        // load experience a velocity spike that triggers the firmware's
        // joint_velocity_violation reflex within ~10ms.
        //
        // Combined mode keeps the internal controller alive:
        //   - Motion generator holds initial position (with low impedance)
        //   - Internal controller provides gravity comp + velocity safety
        //   - Torque callback adds user's desired torques on top
        //
        // Low impedance ensures user torques are the dominant control signal.

        // Set low joint impedance so user torques dominate.
        // The internal controller still runs (gravity comp + velocity limiting)
        // but generates only tiny holding forces.
        try {
            robot_.setJointImpedance({{20, 20, 20, 20, 10, 10, 10}});
            std::cout << "[STREAM] Low impedance set: [20,20,20,20,10,10,10]\n";
        } catch (const franka::Exception& e) {
            std::cerr << "[STREAM] setJointImpedance warning: " << e.what() << "\n";
        }

        std::cout << "[STREAM] Torque streaming started (combined motion+torque)" << std::endl;
        try {
            std::array<double, 7> q_hold{};
            bool first = true;
            bool torque_done = false;  // shared between callbacks (same thread)

            robot_.control(
                // 1st arg: Torque callback — user's additional torques
                [&](const franka::RobotState& rs,
                    franka::Duration period) -> franka::Torques {
                    double dt = period.toSec();
                    if (dt <= 0) dt = 0.001;

                    ++callback_count;

                    update_stream_state_from_robot(rs);

                    std::array<double, 7> target;
                    {
                        std::lock_guard<std::mutex> lock(stream_mutex_);
                        target = stream_torque_target_;
                    }

                    if (!streaming_active_.load()) {
                        target.fill(0.0);
                    }

                    // Rate-limited transition
                    for (int i = 0; i < 7; ++i) {
                        double diff = target[i] - current_tau[i];
                        double max_step = max_dtau * dt;
                        if (diff > max_step) diff = max_step;
                        if (diff < -max_step) diff = -max_step;
                        current_tau[i] += diff;
                    }

                    // Velocity safety damping — extra safeguard on top of the
                    // internal controller's velocity limiting.
                    for (int i = 0; i < 7; ++i) {
                        double limit = DQ_MAX[i] * dq_safety;
                        double abs_dq = std::abs(rs.dq[i]);
                        if (abs_dq > limit) {
                            double excess = abs_dq - limit;
                            double sign = (rs.dq[i] > 0) ? 1.0 : -1.0;
                            current_tau[i] -= Kd_vel * excess * sign;
                        }
                    }

                    if (!streaming_active_.load()) {
                        double sum = 0;
                        for (int i = 0; i < 7; ++i) sum += std::abs(current_tau[i]);
                        if (sum < 1e-4) {
                            torque_done = true;
                            return franka::MotionFinished(
                                franka::Torques({0, 0, 0, 0, 0, 0, 0}));
                        }
                    }

                    return franka::Torques({
                        current_tau[0], current_tau[1], current_tau[2], current_tau[3],
                        current_tau[4], current_tau[5], current_tau[6]});
                },
                // 2nd arg: Motion generator — slowly track actual position.
                // q_hold drifts towards rs.q with ~1s time constant.
                // This avoids snap-back on stop (unlike fixed q_hold),
                // and avoids noise issues (unlike raw rs.q).
                [&](const franka::RobotState& rs,
                    franka::Duration) -> franka::JointPositions {
                    if (first) {
                        q_hold = rs.q_d;
                        first = false;
                    }
                    // Slowly blend q_hold towards actual position.
                    // alpha=0.002 at 1kHz → tau ≈ 0.5s
                    for (int i = 0; i < 7; ++i) {
                        q_hold[i] += 0.002 * (rs.q[i] - q_hold[i]);
                    }
                    if (torque_done) {
                        return franka::MotionFinished(
                            franka::JointPositions(q_hold));
                    }
                    return franka::JointPositions(q_hold);
                },
                /*limit_rate=*/true, /*cutoff_frequency=*/100.0);

            // Normal exit — print summary OUTSIDE the realtime callback
            std::cout << "[STREAM] Torque stream finished (callbacks=" << callback_count << ")" << std::endl;
        } catch (const franka::Exception& e) {
            std::cout << "[STREAM] Torque error (callbacks=" << callback_count << "): " << e.what() << std::endl;
            connection_healthy_ = false;
            try { robot_.automaticErrorRecovery(); connection_healthy_ = true; }
            catch (...) {}
        }

        } // g_robot_mutex released
        streaming_active_ = false;
        stream_state_valid_ = false;
        stream_mode_ = StreamMode::NONE;
        std::cout << "[STREAM] Torque thread exiting" << std::endl;
    }

    // ----------------------------------------------------------------

    void join_streaming_thread_if_needed() {
        if (streaming_thread_.joinable()) {
            streaming_thread_.join();
        }
    }

    // Check and restore connection health before motion commands
    // Returns error message if unhealthy, empty string if OK
    std::string check_connection_health() {
        if (connection_healthy_) {
            return "";  // Already healthy
        }

        // Try to recover connection by reading state
        std::cout << "[HEALTH] Connection marked unhealthy, attempting recovery...\n";

        try {
            // First try automatic error recovery
            robot_.automaticErrorRecovery();
            std::cout << "[HEALTH] automaticErrorRecovery() succeeded\n";
        } catch (const franka::Exception& e) {
            std::cerr << "[HEALTH] automaticErrorRecovery() failed: " << e.what() << "\n";
            // Continue to try readOnce anyway
        }

        try {
            // Try to read state - if this works, connection is restored
            robot_.readOnce();
            connection_healthy_ = true;
            std::cout << "[HEALTH] Connection restored successfully\n";
            return "";
        } catch (const franka::Exception& e) {
            std::cerr << "[HEALTH] readOnce() failed: " << e.what() << "\n";
            return std::string("Robot connection unhealthy: ") + e.what() +
                   ". Please restart daemon or check robot status in Desk UI.";
        }
    }

    msgpack::sbuffer handle_ping() {
        return pack_success_response("pong");
    }

    msgpack::sbuffer handle_get_state() {
        RobotStateSnapshot state = get_current_state();
        return pack_state_response(state);
    }

    msgpack::sbuffer handle_move_to_qpos(const std::map<std::string, msgpack::object>& msg) {
        // Stop any active streaming before position move
        if (stream_mode_ != StreamMode::NONE) {
            stop_streaming_internal();
        }

        std::lock_guard<std::mutex> lock(g_robot_mutex);

        // Check connection health before attempting motion
        std::string health_error = check_connection_health();
        if (!health_error.empty()) {
            return pack_error_response(health_error);
        }

        // Parse parameters
        std::array<double, 7> qpos;
        auto it = msg.find("qpos");
        if (it == msg.end()) {
            return pack_error_response("Missing 'qpos' field");
        }
        std::vector<double> qpos_vec = it->second.as<std::vector<double>>();
        if (qpos_vec.size() != 7) {
            return pack_error_response("qpos must have 7 elements");
        }
        std::copy(qpos_vec.begin(), qpos_vec.end(), qpos.begin());

        double speed_scale = 0.15;
        it = msg.find("speed_scale");
        if (it != msg.end()) speed_scale = it->second.as<double>();

        double collision_torque = 70.0;
        it = msg.find("collision_torque");
        if (it != msg.end()) collision_torque = it->second.as<double>();

        double collision_force = 70.0;
        it = msg.find("collision_force");
        if (it != msg.end()) collision_force = it->second.as<double>();

        std::cout << "[CMD] move_to_qpos: speed_scale=" << speed_scale << "\n";

        try {
            // Set collision behavior
            franka_servo::set_conservative_collision_behavior(
                robot_, collision_torque, collision_force);

            // Load servo params from unified Python config (single source of truth)
            // This ensures sim-real parity: Python motion_config.py -> JSON -> C++
            robot_params::RobotParams config = robot_params::loadFromJSON(CONFIG_PATH);
            franka_servo::JointServoParams params = config.joint;

            // Override only the per-command parameters from Python
            params.dq_cap = speed_scale;      // From Python command message
            params.timeout_sec = 25.0;        // Keep < Python client timeout (30s)

            // Log loaded params for verification
            std::cout << "[CONFIG] JointServoParams loaded from " << CONFIG_PATH << ":\n"
                      << "         Kp_joint=" << params.Kp_joint << ", dq_cap=" << params.dq_cap << " rad/s (from Python)\n"
                      << "         ramp_time=" << params.ramp_time << "s, q_tol=" << params.q_tol << " rad\n";

            franka_servo::servo_joint_position(robot_, qpos, params);

            return pack_success_response("move_to_qpos completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] move_to_qpos error: " << e.what() << "\n";

            // Mark connection as unhealthy - next command will attempt recovery
            connection_healthy_ = false;

            // Attempt recovery and LOG the result (don't silently swallow)
            try {
                robot_.automaticErrorRecovery();
                std::cout << "[CMD] automaticErrorRecovery() succeeded after move_to_qpos error\n";
                connection_healthy_ = true;
            } catch (const franka::Exception& recovery_error) {
                std::cerr << "[CMD] automaticErrorRecovery() failed: " << recovery_error.what() << "\n";
            }

            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_move_to_cartesian(const std::map<std::string, msgpack::object>& msg) {
        // Stop any active streaming before position move
        if (stream_mode_ != StreamMode::NONE) {
            stop_streaming_internal();
        }

        std::lock_guard<std::mutex> lock(g_robot_mutex);

        // Check connection health before attempting motion
        std::string health_error = check_connection_health();
        if (!health_error.empty()) {
            return pack_error_response(health_error);
        }

        // Parse position
        auto it = msg.find("position");
        if (it == msg.end()) {
            return pack_error_response("Missing 'position' field");
        }
        std::vector<double> pos_vec = it->second.as<std::vector<double>>();
        if (pos_vec.size() != 3) {
            return pack_error_response("position must have 3 elements");
        }
        Eigen::Vector3d target_pos(pos_vec[0], pos_vec[1], pos_vec[2]);

        // Parse orientation (9-element row-major rotation matrix)
        it = msg.find("orientation");
        if (it == msg.end()) {
            return pack_error_response("Missing 'orientation' field");
        }
        std::vector<double> orn_vec = it->second.as<std::vector<double>>();
        if (orn_vec.size() != 9) {
            return pack_error_response("orientation must have 9 elements (row-major 3x3)");
        }
        Eigen::Matrix3d target_R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                target_R(i, j) = orn_vec[i * 3 + j];
            }
        }

        double move_speed = 0.05;
        it = msg.find("move_speed");
        if (it != msg.end()) move_speed = it->second.as<double>();

        double collision_torque = 70.0;
        it = msg.find("collision_torque");
        if (it != msg.end()) collision_torque = it->second.as<double>();

        double collision_force = 70.0;
        it = msg.find("collision_force");
        if (it != msg.end()) collision_force = it->second.as<double>();

        std::cout << "[CMD] move_to_cartesian: pos=[" << target_pos.x() << ", "
                  << target_pos.y() << ", " << target_pos.z() << "], speed="
                  << move_speed << "\n";

        try {
            franka_servo::set_conservative_collision_behavior(
                robot_, collision_torque, collision_force);

            // Load servo params from unified Python config (single source of truth)
            // This ensures sim-real parity: Python motion_config.py -> JSON -> C++
            robot_params::RobotParams config = robot_params::loadFromJSON(CONFIG_PATH);
            franka_servo::CartesianServoParams params = config.cartesian;

            // Override only the per-command parameters from Python
            params.v_cap = move_speed;        // From Python command message
            params.timeout_sec = 25.0;        // Keep < Python client timeout (30s)

            // Log loaded params for verification
            std::cout << "[CONFIG] CartesianServoParams loaded from " << CONFIG_PATH << ":\n"
                      << "         Kp_lin=" << params.Kp_lin << ", Kp_ang=" << params.Kp_ang << "\n"
                      << "         v_cap=" << params.v_cap << " m/s (from Python), w_cap=" << params.w_cap << " rad/s\n"
                      << "         ramp_time=" << params.ramp_time << "s, pos_tol=" << params.pos_tol << "m\n";

            franka_servo::servo_cartesian_pose(robot_, target_pos, target_R, params);

            return pack_success_response("move_to_cartesian completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] move_to_cartesian error: " << e.what() << "\n";

            // Mark connection as unhealthy - next command will attempt recovery
            connection_healthy_ = false;

            // Attempt recovery and LOG the result (don't silently swallow)
            try {
                robot_.automaticErrorRecovery();
                std::cout << "[CMD] automaticErrorRecovery() succeeded after move_to_cartesian error\n";
                connection_healthy_ = true;
            } catch (const franka::Exception& recovery_error) {
                std::cerr << "[CMD] automaticErrorRecovery() failed: " << recovery_error.what() << "\n";
            }

            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_cartesian_velocity(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("twist");
        if (it == msg.end()) return pack_error_response("Missing 'twist' field");
        std::vector<double> twist_vec = it->second.as<std::vector<double>>();
        if (twist_vec.size() != 6) return pack_error_response("twist must have 6 elements");

        // Update target
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            for (int i = 0; i < 6; ++i) stream_twist_target_[i] = twist_vec[i];
        }

        // If already streaming cartesian velocity, just update
        if (stream_mode_ == StreamMode::CART_VEL && streaming_active_.load()) {
            return pack_success_response("velocity_updated");
        }

        if (stream_mode_ != StreamMode::NONE) {
            stop_streaming_internal();
        }

        {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            std::string health_error = check_connection_health();
            if (!health_error.empty()) return pack_error_response(health_error);
        }

        std::cout << "[CMD] set_cartesian_velocity: starting streaming mode\n";
        join_streaming_thread_if_needed();
        stream_mode_ = StreamMode::CART_VEL;
        streaming_active_ = true;
        streaming_thread_ = std::thread(&CommandHandler::run_cartesian_velocity_stream, this);

        return pack_success_response("streaming_started");
    }

    msgpack::sbuffer handle_open_gripper(const std::map<std::string, msgpack::object>& msg) {
        double width = 0.08;
        auto it = msg.find("width");
        if (it != msg.end()) width = it->second.as<double>();

        double speed = 0.05;
        it = msg.find("speed");
        if (it != msg.end()) speed = it->second.as<double>();

        std::cout << "[CMD] open_gripper: width=" << width << "m, speed=" << speed << "m/s\n";

        if (!gripper_) {
            return pack_error_response("Gripper not connected");
        }

        try {
            gripper_->stop();
            bool ok = gripper_->move(width, speed);
            last_gripper_width_ = width;
            is_grasping_ = false;

            if (ok) {
                return pack_success_response("open_gripper completed");
            } else {
                return pack_error_response("Gripper move returned false");
            }
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] open_gripper error: " << e.what() << "\n";
            return pack_error_response(std::string("Gripper error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_grasp(const std::map<std::string, msgpack::object>& msg) {
        double force = 60.0;
        auto it = msg.find("force");
        if (it != msg.end()) force = it->second.as<double>();

        double speed = 0.05;
        it = msg.find("speed");
        if (it != msg.end()) speed = it->second.as<double>();

        double eps_inner = 0.005;
        it = msg.find("eps_inner");
        if (it != msg.end()) eps_inner = it->second.as<double>();

        double eps_outer = 0.005;
        it = msg.find("eps_outer");
        if (it != msg.end()) eps_outer = it->second.as<double>();

        std::cout << "[CMD] grasp: force=" << force << "N, speed=" << speed << "m/s\n";

        if (!gripper_) {
            return pack_error_response("Gripper not connected");
        }

        try {
            gripper_->stop();
            bool grasped = gripper_->grasp(0.0, speed, force, eps_inner, eps_outer);

            franka::GripperState gs = gripper_->readOnce();
            last_gripper_width_ = gs.width;
            is_grasping_ = gs.is_grasped;

            bool object_detected = (gs.width > 0.002);  // Width > 2mm = object present

            std::cout << "[CMD] grasp result: width=" << gs.width
                      << "m, is_grasped=" << (gs.is_grasped ? "true" : "false")
                      << ", object_detected=" << (object_detected ? "true" : "false") << "\n";

            return pack_grasp_response(true, object_detected, gs.width);
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] grasp error: " << e.what() << "\n";
            return pack_grasp_response(false, false, 0.0);
        }
    }

    msgpack::sbuffer handle_set_joint_velocity(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("dq");
        if (it == msg.end()) return pack_error_response("Missing 'dq' field");
        std::vector<double> dq_vec = it->second.as<std::vector<double>>();
        if (dq_vec.size() != 7) return pack_error_response("dq must have 7 elements");

        int duration_ms = 0;  // default: streaming
        it = msg.find("duration_ms");
        if (it != msg.end()) duration_ms = it->second.as<int>();

        // Update target
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            for (int i = 0; i < 7; ++i) stream_dq_target_[i] = dq_vec[i];
        }

        // If already streaming joint velocity, just update target
        if (stream_mode_ == StreamMode::JOINT_VEL && streaming_active_.load()) {
            return pack_success_response("velocity_updated");
        }

        // Stop any existing different-mode stream
        if (stream_mode_ != StreamMode::NONE) {
            stop_streaming_internal();
        }

        // Start new streaming thread
        {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            std::string health_error = check_connection_health();
            if (!health_error.empty()) return pack_error_response(health_error);
        }

        std::cout << "[CMD] set_joint_velocity: starting streaming mode\n";
        join_streaming_thread_if_needed();
        stream_mode_ = StreamMode::JOINT_VEL;  // Set BEFORE streaming_active_ so get_current_state sees it
        streaming_active_ = true;
        streaming_thread_ = std::thread(&CommandHandler::run_joint_velocity_stream, this);

        return pack_success_response("streaming_started");
    }

    msgpack::sbuffer handle_set_torques(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("torques");
        if (it == msg.end()) return pack_error_response("Missing 'torques' field");
        std::vector<double> tau_vec = it->second.as<std::vector<double>>();
        if (tau_vec.size() != 7) return pack_error_response("torques must have 7 elements");

        // Update target
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            for (int i = 0; i < 7; ++i) stream_torque_target_[i] = tau_vec[i];
        }

        // If already streaming torque, just update
        if (stream_mode_ == StreamMode::TORQUE && streaming_active_.load()) {
            return pack_success_response("torque_updated");
        }

        if (stream_mode_ != StreamMode::NONE) {
            stop_streaming_internal();
        }

        {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            std::string health_error = check_connection_health();
            if (!health_error.empty()) return pack_error_response(health_error);

            // Relax collision thresholds for direct torque control.
            // In torque mode, commanded torques can look like "collisions"
            // to the safety system.
            franka_servo::set_conservative_collision_behavior(robot_, 100.0, 100.0);
        }

        std::cout << "[CMD] set_torques: starting streaming mode\n";
        join_streaming_thread_if_needed();
        stream_mode_ = StreamMode::TORQUE;
        streaming_active_ = true;
        streaming_thread_ = std::thread(&CommandHandler::run_torque_stream, this);

        // Wait briefly to confirm the stream actually started.
        // Without this, Python receives "streaming_started" but the stream may
        // have already crashed (e.g. joint_velocity_violation), causing a silent
        // failure cascade where every 50ms a new thread is spawned and dies.
        for (int i = 0; i < 100; ++i) {  // up to 100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (stream_state_valid_.load()) {
                return pack_success_response("streaming_started");
            }
            if (!streaming_active_.load() && stream_mode_.load() == StreamMode::NONE) {
                // Stream thread died — report the failure to Python
                join_streaming_thread_if_needed();
                return pack_error_response(
                    "Torque stream failed immediately (likely joint_velocity_violation). "
                    "Try error_recovery first, or check Desk for reflex errors.");
            }
        }

        return pack_success_response("streaming_started");
    }

    msgpack::sbuffer handle_set_joint_impedance(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("K_theta");
        if (it == msg.end()) return pack_error_response("Missing 'K_theta' field");
        std::vector<double> K_vec = it->second.as<std::vector<double>>();
        if (K_vec.size() != 7) return pack_error_response("K_theta must have 7 elements");

        std::array<double, 7> K_theta{};
        for (int i = 0; i < 7; ++i) K_theta[i] = K_vec[i];

        std::cout << "[CMD] set_joint_impedance: K=[" << K_theta[0];
        for (int i = 1; i < 7; ++i) std::cout << ", " << K_theta[i];
        std::cout << "]\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.setJointImpedance(K_theta);
            return pack_success_response("set_joint_impedance completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_joint_impedance error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_cartesian_impedance(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("K_x");
        if (it == msg.end()) return pack_error_response("Missing 'K_x' field");
        std::vector<double> K_vec = it->second.as<std::vector<double>>();
        if (K_vec.size() != 6) return pack_error_response("K_x must have 6 elements");

        std::array<double, 6> K_x{};
        for (int i = 0; i < 6; ++i) K_x[i] = K_vec[i];

        std::cout << "[CMD] set_cartesian_impedance: K=[" << K_x[0];
        for (int i = 1; i < 6; ++i) std::cout << ", " << K_x[i];
        std::cout << "]\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.setCartesianImpedance(K_x);
            return pack_success_response("set_cartesian_impedance completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_cartesian_impedance error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_collision_behavior(const std::map<std::string, msgpack::object>& msg) {
        auto it_tl = msg.find("torque_lower");
        auto it_tu = msg.find("torque_upper");
        auto it_fl = msg.find("force_lower");
        auto it_fu = msg.find("force_upper");

        if (it_tl == msg.end() || it_tu == msg.end() ||
            it_fl == msg.end() || it_fu == msg.end()) {
            return pack_error_response("Missing collision behavior fields");
        }

        std::vector<double> tl = it_tl->second.as<std::vector<double>>();
        std::vector<double> tu = it_tu->second.as<std::vector<double>>();
        std::vector<double> fl = it_fl->second.as<std::vector<double>>();
        std::vector<double> fu = it_fu->second.as<std::vector<double>>();

        if (tl.size() != 7 || tu.size() != 7 || fl.size() != 6 || fu.size() != 6) {
            return pack_error_response("Invalid collision behavior array sizes");
        }

        std::array<double, 7> a_tl{}, a_tu{};
        std::array<double, 6> a_fl{}, a_fu{};
        for (int i = 0; i < 7; ++i) { a_tl[i] = tl[i]; a_tu[i] = tu[i]; }
        for (int i = 0; i < 6; ++i) { a_fl[i] = fl[i]; a_fu[i] = fu[i]; }

        std::cout << "[CMD] set_collision_behavior\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            franka_servo::detail::set_collision_behavior_impl(
                robot_, a_tl, a_tu, a_tl, a_tu, a_fl, a_fu, a_fl, a_fu, 0);
            return pack_success_response("set_collision_behavior completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_collision_behavior error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_load(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("mass");
        if (it == msg.end()) return pack_error_response("Missing 'mass' field");
        double mass = it->second.as<double>();

        it = msg.find("F_x_Cload");
        if (it == msg.end()) return pack_error_response("Missing 'F_x_Cload' field");
        std::vector<double> com_vec = it->second.as<std::vector<double>>();
        if (com_vec.size() != 3) return pack_error_response("F_x_Cload must have 3 elements");
        std::array<double, 3> F_x_Cload{};
        for (int i = 0; i < 3; ++i) F_x_Cload[i] = com_vec[i];

        it = msg.find("load_inertia");
        if (it == msg.end()) return pack_error_response("Missing 'load_inertia' field");
        std::vector<double> inertia_vec = it->second.as<std::vector<double>>();
        if (inertia_vec.size() != 9) return pack_error_response("load_inertia must have 9 elements");
        std::array<double, 9> load_inertia{};
        for (int i = 0; i < 9; ++i) load_inertia[i] = inertia_vec[i];

        std::cout << "[CMD] set_load: mass=" << mass << "kg\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.setLoad(mass, F_x_Cload, load_inertia);
            return pack_success_response("set_load completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_load error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_ee(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("NE_T_EE");
        if (it == msg.end()) return pack_error_response("Missing 'NE_T_EE' field");
        std::vector<double> ee_vec = it->second.as<std::vector<double>>();
        if (ee_vec.size() != 16) return pack_error_response("NE_T_EE must have 16 elements");

        std::array<double, 16> NE_T_EE{};
        for (int i = 0; i < 16; ++i) NE_T_EE[i] = ee_vec[i];

        std::cout << "[CMD] set_ee\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.setEE(NE_T_EE);
            return pack_success_response("set_ee completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_ee error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_set_guiding_mode(const std::map<std::string, msgpack::object>& msg) {
        auto it = msg.find("guiding_axes");
        if (it == msg.end()) return pack_error_response("Missing 'guiding_axes' field");
        std::vector<bool> axes_vec = it->second.as<std::vector<bool>>();
        if (axes_vec.size() != 6) return pack_error_response("guiding_axes must have 6 elements");

        std::array<bool, 6> guiding_axes{};
        for (int i = 0; i < 6; ++i) guiding_axes[i] = axes_vec[i];

        bool nullspace = false;
        it = msg.find("nullspace");
        if (it != msg.end()) nullspace = it->second.as<bool>();

        std::cout << "[CMD] set_guiding_mode: nullspace=" << (nullspace ? "true" : "false") << "\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.setGuidingMode(guiding_axes, nullspace);
            return pack_success_response("set_guiding_mode completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] set_guiding_mode error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_error_recovery() {
        std::cout << "[CMD] error_recovery\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.automaticErrorRecovery();
            connection_healthy_ = true;
            return pack_success_response("error_recovery completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] error_recovery error: " << e.what() << "\n";
            return pack_error_response(std::string("libfranka error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_stop() {
        // First stop any streaming
        stop_streaming_internal();

        std::cout << "[CMD] stop: attempting emergency stop\n";

        try {
            std::lock_guard<std::mutex> lock(g_robot_mutex);
            robot_.stop();
            if (gripper_) gripper_->stop();
            return pack_success_response("stop completed");
        } catch (const franka::Exception& e) {
            std::cerr << "[CMD] stop error: " << e.what() << "\n";
            return pack_error_response(std::string("Stop error: ") + e.what());
        }
    }

    msgpack::sbuffer handle_stop_streaming() {
        std::cout << "[CMD] stop_streaming\n";
        // Signal thread to ramp to zero, but DON'T block on join().
        // The streaming thread will ramp down and exit on its own.
        // stream_mode_ stays set until thread exits (thread clears it),
        // so get_current_state() won't call readOnce() prematurely.
        streaming_active_ = false;
        return pack_success_response("streaming stopped");
    }
};

// =============================================================================
// STATE STREAMING THREAD
// =============================================================================

void state_streaming_thread(zmq::socket_t& pub_socket, CommandHandler& handler) {
    using clock = std::chrono::steady_clock;
    const auto stream_interval = std::chrono::milliseconds(10);  // 100 Hz

    while (g_running) {
        auto start = clock::now();

        try {
            RobotStateSnapshot state = handler.get_current_state();
            msgpack::sbuffer buffer = pack_state_response(state);

            zmq::message_t message(buffer.data(), buffer.size());
            pub_socket.send(message, zmq::send_flags::dontwait);
        } catch (const zmq::error_t& e) {
            if (e.num() != EAGAIN) {
                std::cerr << "[STREAM] ZeroMQ error: " << e.what() << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[STREAM] Error: " << e.what() << "\n";
        }

        auto elapsed = clock::now() - start;
        auto sleep_time = stream_interval - elapsed;
        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-ip> [--command_port PORT] [--state_port PORT]\n";
        return 1;
    }

    std::string robot_ip = argv[1];
    int command_port = servo_protocol::DEFAULT_COMMAND_PORT;
    int state_port = servo_protocol::DEFAULT_STATE_PORT;

    // Parse optional arguments
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        if (std::strcmp(argv[i], "--command_port") == 0) {
            command_port = std::stoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--state_port") == 0) {
            state_port = std::stoi(argv[i + 1]);
        }
    }

    std::cout << "==========================================\n";
    std::cout << "FrankaDaemon\n";
    std::cout << "==========================================\n";
    std::cout << "Robot IP:     " << robot_ip << "\n";
    std::cout << "Command port: " << command_port << "\n";
    std::cout << "State port:   " << state_port << "\n";
    std::cout << "==========================================\n";

    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Connect to robot
        std::cout << "[INIT] Connecting to robot...\n";
        franka::Robot robot(robot_ip);
        std::cout << "[INIT] Robot connected.\n";

        // Connect to gripper (optional — may not be available)
        std::unique_ptr<franka::Gripper> gripper;
        try {
            gripper = std::make_unique<franka::Gripper>(robot_ip);
            std::cout << "[INIT] Gripper connected.\n";
        } catch (const franka::Exception& e) {
            std::cerr << "[INIT] Gripper not available: " << e.what() << "\n";
            std::cerr << "[INIT] Continuing without gripper.\n";
        }

        std::cout << "[INIT] Setting collision behavior...\n";
        franka_servo::set_conservative_collision_behavior(robot, 70.0, 70.0);

        // Initialize ZeroMQ
        std::cout << "[INIT] Setting up ZeroMQ sockets...\n";
        zmq::context_t context(1);

        // REP socket for commands
        zmq::socket_t rep_socket(context, zmq::socket_type::rep);
        rep_socket.bind(servo_protocol::tcp_endpoint(command_port));
        rep_socket.set(zmq::sockopt::rcvtimeo, 100);  // 100ms timeout

        // PUB socket for state streaming
        zmq::socket_t pub_socket(context, zmq::socket_type::pub);
        pub_socket.bind(servo_protocol::tcp_endpoint(state_port));

        CommandHandler handler(robot, gripper.get());

        // Start state streaming thread
        std::thread stream_thread(state_streaming_thread,
                                  std::ref(pub_socket),
                                  std::ref(handler));

        std::cout << "[READY] Daemon ready, waiting for commands...\n";

        // Main command loop
        while (g_running) {
            zmq::message_t request;

            try {
                auto result = rep_socket.recv(request, zmq::recv_flags::none);
                if (!result) continue;  // Timeout

                // Unpack message
                msgpack::object_handle oh = msgpack::unpack(
                    static_cast<const char*>(request.data()),
                    request.size()
                );
                msgpack::object obj = oh.get();

                // Handle command
                msgpack::sbuffer response = handler.handle_command(obj);

                // Send response
                zmq::message_t reply(response.data(), response.size());
                rep_socket.send(reply, zmq::send_flags::none);

            } catch (const zmq::error_t& e) {
                if (e.num() == EAGAIN) continue;  // Timeout
                std::cerr << "[MAIN] ZeroMQ error: " << e.what() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[MAIN] Error: " << e.what() << "\n";

                // Send error response
                msgpack::sbuffer err = pack_error_response(e.what());
                zmq::message_t reply(err.data(), err.size());
                try {
                    rep_socket.send(reply, zmq::send_flags::none);
                } catch (...) {}
            }
        }

        std::cout << "[SHUTDOWN] Stopping state stream...\n";
        stream_thread.join();

        std::cout << "[SHUTDOWN] Closing sockets...\n";
        rep_socket.close();
        pub_socket.close();
        context.close();

        std::cout << "[SHUTDOWN] Complete.\n";
        return 0;

    } catch (const franka::Exception& e) {
        std::cerr << "[FATAL] libfranka exception: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << "\n";
        return 3;
    }
}
