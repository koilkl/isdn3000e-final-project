#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

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
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    uint8_t target_cell_id = 255;
    for (size_t i = 0; i < request->snapshot.legal_actions.size(); ++i) {
      if (request->snapshot.legal_actions[i] == 1) {
        target_cell_id = static_cast<uint8_t>(i);
        break;
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
