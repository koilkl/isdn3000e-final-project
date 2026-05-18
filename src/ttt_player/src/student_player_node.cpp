#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#if defined(TTT_PLAYER_HAS_TORCH)
#include <torch/script.h>
#endif

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

const std::vector<std::pair<uint8_t, uint8_t>> kScriptedMoves = {
    {6, 2},
    {7, 4},
    {8, 6},
};

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}

std::optional<uint8_t> choose_cell_rule_based(
    const ttt_interfaces::msg::GameSnapshot &snapshot,
    uint8_t player_id) {
  const auto my_mark = player_id == 0 ? ttt_interfaces::msg::GameSnapshot::PLAYER_0
                                      : ttt_interfaces::msg::GameSnapshot::PLAYER_1;
  const auto opp_mark = player_id == 0 ? ttt_interfaces::msg::GameSnapshot::PLAYER_1
                                       : ttt_interfaces::msg::GameSnapshot::PLAYER_0;

  auto is_legal = [&](uint8_t cell) {
    return cell < snapshot.legal_actions.size() && snapshot.legal_actions[cell] == 1;
  };

  auto would_win = [&](uint8_t mark, uint8_t cell) {
    if (!is_legal(cell)) {
      return false;
    }
    std::array<uint8_t, 9> board = snapshot.board;
    board[cell] = mark;
    constexpr std::array<std::array<uint8_t, 3>, 8> lines = {{{{0, 1, 2}},
                                                              {{3, 4, 5}},
                                                              {{6, 7, 8}},
                                                              {{0, 3, 6}},
                                                              {{1, 4, 7}},
                                                              {{2, 5, 8}},
                                                              {{0, 4, 8}},
                                                              {{2, 4, 6}}}};
    for (const auto &line : lines) {
      if (board[line[0]] == mark && board[line[1]] == mark && board[line[2]] == mark) {
        return true;
      }
    }
    return false;
  };

  for (uint8_t cell = 0; cell < 9; ++cell) {
    if (would_win(my_mark, cell)) {
      return cell;
    }
  }

  for (uint8_t cell = 0; cell < 9; ++cell) {
    if (would_win(opp_mark, cell)) {
      return cell;
    }
  }

  if (is_legal(4)) {
    return 4;
  }

  constexpr std::array<uint8_t, 4> corners = {0, 2, 6, 8};
  for (auto cell : corners) {
    if (is_legal(cell)) {
      return cell;
    }
  }

  constexpr std::array<uint8_t, 4> edges = {1, 3, 5, 7};
  for (auto cell : edges) {
    if (is_legal(cell)) {
      return cell;
    }
  }

  return std::nullopt;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    // Use ament_index_cpp to find the model path relative to the installed package share directory
    std::string default_model_path;
    try {
      std::string package_share_directory = ament_index_cpp::get_package_share_directory("ttt_player");
      default_model_path = package_share_directory + "/resource/ttt_rl_model.pt";
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "Failed to get package share directory: %s", e.what());
      default_model_path = "ttt_rl_model.pt"; // Fallback
    }

    this->declare_parameter<std::string>("model_path", default_model_path);

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();
    
#if defined(TTT_PLAYER_HAS_TORCH)
    std::string model_path = this->get_parameter("model_path").as_string();
    try {
      rl_model_ = torch::jit::load(model_path);
      rl_model_loaded_ = true;
      RCLCPP_INFO(this->get_logger(), "RL model loaded successfully from %s", model_path.c_str());
    } catch (const c10::Error &e) {
      rl_model_loaded_ = false;
      RCLCPP_ERROR(this->get_logger(), "Error loading the RL model from %s: %s", model_path.c_str(), e.what());
    }
#else
    RCLCPP_INFO(this->get_logger(), "Built without torch support. Using rule-based policy.");
#endif

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  void try_register() {
    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ----------------------------------------------------------------
  // IK helpers
  // ----------------------------------------------------------------

  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {
    // TODO(student): Call the MoveIt `/compute_ik` service here.
    // Suggested steps:
    // 1. Create a `moveit_msgs::srv::GetPositionIK::Request`.
    // 2. Set `group_name = "panda_arm"`.
    // 3. Fill the seed joint state with the provided `seed_positions`.
    // 4. Set the target pose in frame `panda_link0`.
    // 5. Send the request through `ik_client_` and wait for the response.
    // 6. Extract the 7 Panda arm joints from the solution and return them.
    // 7. Return `std::nullopt` if IK times out or fails.
    //
    // The dummy return below keeps the starter code buildable, but it does not
    // solve IK. Students should replace it with a real implementation.
    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name = "panda_arm";
    request->ik_request.robot_state.joint_state.name = panda_joint_names();
    request->ik_request.robot_state.joint_state.position = seed_positions;
    request->ik_request.pose_stamped.header.frame_id = "panda_link0";
    request->ik_request.pose_stamped.pose = target_pose;
    request->ik_request.timeout.sec = 5;
    request->ik_request.avoid_collisions = false;

    if (!ik_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "IK service not available");
      return std::nullopt;
    }

    auto future = ik_client_->async_send_request(request);
    if (future.wait_for(10s) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "IK service timeout");
      return std::nullopt;
    }

    auto response = future.get();
    if (response->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "IK failed with error code %d", response->error_code.val);
      return std::nullopt;
    }

    std::vector<double> result;
    result.reserve(7);
    const auto& names = response->solution.joint_state.name;
    const auto& positions = response->solution.joint_state.position;
    for (const auto& target_name : panda_joint_names()) {
      auto it = std::find(names.begin(), names.end(), target_name);
      if (it != names.end()) {
        auto index = std::distance(names.begin(), it);
        result.push_back(positions[index]);
      } else {
        return std::nullopt;
      }
    }
    return result;
  }

  // ----------------------------------------------------------------
  // Turn planning
  // ----------------------------------------------------------------

  void handle_plan_turn(
       // TODO(student): Implement your turn-planning logic here.
    // Suggested structure:
    // 1. Choose a legal `(piece_id, cell_id)` pair from `request->snapshot`.
    // 2. Look up the current pose of the chosen stock piece.
    // 3. Look up the target board cell pose from `request->layout.cell_poses`.
    // 4. Convert those TCP targets into `panda_link8` poses using
    //    `link8_pose_from_tcp_target(...)`.
    // 5. Call `compute_ik(...)` for the pick target and place target.
    // 6. Build the four required trajectories:
    //      - home_to_pick
    //      - pick_to_home
    //      - home_to_place
    //      - place_to_home
    // 7. Fill `response->plan` and set `response->accepted = true` on success.
    //
    // The fallback below intentionally rejects every turn. This keeps the
    // starter repository buildable while making it clear that students must
    // implement their own planner.
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response)
       {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    uint8_t target_cell_id = 255;

#if defined(TTT_PLAYER_HAS_TORCH)
    if (rl_model_loaded_) {
      std::vector<float> board_state(9, 0.0F);
      for (size_t i = 0; i < 9; ++i) {
        board_state[i] = static_cast<float>(request->snapshot.board[i]);
      }
      torch::Tensor state_tensor = torch::from_blob(board_state.data(), {1, 9}, torch::kFloat32).clone();

      try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(state_tensor);
        torch::Tensor output = rl_model_.forward(inputs).toTensor();

        float max_score = -std::numeric_limits<float>::infinity();
        auto output_accessor = output.accessor<float, 2>();
        for (size_t i = 0; i < 9; ++i) {
          if (request->snapshot.legal_actions[i] == 1) {
            float score = output_accessor[0][i];
            if (score > max_score) {
              max_score = score;
              target_cell_id = static_cast<uint8_t>(i);
            }
          }
        }
        RCLCPP_INFO(this->get_logger(), "RL model selected cell %d", target_cell_id);
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "RL model inference failed: %s", e.what());
      }
    }
#endif

    if (target_cell_id == 255) {
      const auto cell = choose_cell_rule_based(request->snapshot, player_id_);
      if (cell) {
        target_cell_id = *cell;
      } else {
        for (size_t i = 0; i < request->snapshot.legal_actions.size(); ++i) {
          if (request->snapshot.legal_actions[i] == 1) {
            target_cell_id = static_cast<uint8_t>(i);
            break;
          }
        }
      }
    }

    uint8_t target_piece_id = 255;
    for (const auto& piece : request->snapshot.pieces) {
      if (piece.owner == player_id_ && piece.available) {
        target_piece_id = piece.piece_id;
        break;
      }
    }

    if (target_cell_id == 255 || target_piece_id == 255) {
      response->accepted = false;
      response->message = "No legal moves available.";
      return;
    }

    geometry_msgs::msg::Pose piece_pose;
    try {
      piece_pose = find_piece_pose(request->snapshot, request->layout, target_piece_id);
    } catch (const std::exception& e) {
      response->accepted = false;
      response->message = e.what();
      return;
    }

    geometry_msgs::msg::Pose cell_pose = request->layout.cell_poses[target_cell_id];

    auto pick_target = link8_pose_from_tcp_target(piece_pose.position.x, piece_pose.position.y, piece_pose.position.z);
    auto place_target = link8_pose_from_tcp_target(cell_pose.position.x, cell_pose.position.y, cell_pose.position.z);

    auto pick_goal = compute_ik(pick_target, kHomePositions);
    if (!pick_goal) {
      response->accepted = false;
      response->message = "IK failed for pick target.";
      return;
    }

    auto place_goal = compute_ik(place_target, kHomePositions);
    if (!place_goal) {
      response->accepted = false;
      response->message = "IK failed for place target.";
      return;
    }

    ttt_interfaces::msg::TurnPlan plan;
    plan.match_id = request->match_id;
    plan.turn_index = request->turn_index;
    plan.player_id = request->player_id;
    plan.piece_id = target_piece_id;
    plan.cell_id = target_cell_id;
    plan.home_to_pick = make_three_point_trajectory(kHomePositions, *pick_goal, 1.5);
    plan.pick_to_home = make_three_point_trajectory(*pick_goal, kHomePositions, 1.5);
    plan.home_to_place = make_three_point_trajectory(kHomePositions, *place_goal, 1.5);
    plan.place_to_home = make_three_point_trajectory(*place_goal, kHomePositions, 1.5);

    response->plan = plan;
    response->accepted = true;
    response->message = "Plan accepted.";
  }

  static geometry_msgs::msg::Pose find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      const ttt_interfaces::msg::WorkspaceLayout &layout,
      uint8_t piece_id) {
    for (const auto& piece : snapshot.pieces) {
      if (piece.piece_id == piece_id) {
        return piece.pose;
      }
    }
    for (const auto& piece : layout.initial_pieces) {
      if (piece.piece_id == piece_id) {
        return piece.pose;
      }
    }
    throw std::runtime_error("Piece not found");
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};

#if defined(TTT_PLAYER_HAS_TORCH)
  torch::jit::script::Module rl_model_;
  bool rl_model_loaded_{false};
#endif

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
