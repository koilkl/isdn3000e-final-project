from __future__ import annotations

import os
import tempfile

import numpy as np
from moveit_msgs.msg import RobotTrajectory
import rclpy
from rclpy.node import Node
from yourdfpy import URDF

from ttt_interfaces.msg import ExecutionResult
from ttt_interfaces.srv import ReviewTurn


EXPECTED_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = {
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

LINK8_TCP_Z_OFFSET = 0.1034


def _load_panda_urdf(load_meshes: bool, build_scene_graph: bool) -> URDF:
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory("moveit_resources_panda_description")
    except Exception:
        share_dir = "/opt/ros/humble/share/moveit_resources_panda_description"

    urdf_path = os.path.join(share_dir, "urdf", "panda.urdf")
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    urdf_text = urdf_text.replace(
        "package://moveit_resources_panda_description/", f"{share_dir}/"
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, encoding="utf-8"
        ) as f:
            f.write(urdf_text)
            tmp_path = f.name
        return URDF.load(
            tmp_path,
            load_meshes=load_meshes,
            build_scene_graph=build_scene_graph,
            load_collision_meshes=False,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class TicTacToeRefereeNode(Node):
    def __init__(self) -> None:
        super().__init__("ttt_referee")
        self.declare_parameter("endpoint_position_tolerance", 0.07)
        self.endpoint_position_tolerance = float(
            self.get_parameter("endpoint_position_tolerance").value
        )
        self.urdf = _load_panda_urdf(load_meshes=False, build_scene_graph=True)
        self.review_srv = self.create_service(
            ReviewTurn, "/ttt/review_turn", self._on_review
        )

    def _on_review(
        self, request: ReviewTurn.Request, response: ReviewTurn.Response
    ) -> ReviewTurn.Response:
        result = ExecutionResult()
        snapshot = request.snapshot
        piece_lookup = {piece.piece_id: piece for piece in snapshot.pieces}

        if (
            int(request.plan.cell_id) < 0
            or int(request.plan.cell_id) > 8
            or snapshot.legal_actions[int(request.plan.cell_id)] != 1
        ):
            result.success = False
            result.failure_reason = ExecutionResult.REASON_INVALID_CELL
            result.message = f"Cell {request.plan.cell_id} is not legal."
            response.result = result
            return response

        piece = piece_lookup.get(int(request.plan.piece_id))
        if (
            piece is None
            or not piece.available
            or piece.owner != request.plan.player_id
        ):
            result.success = False
            result.failure_reason = ExecutionResult.REASON_INVALID_PIECE
            result.message = f"Piece {request.plan.piece_id} is invalid for player_{request.plan.player_id}."
            response.result = result
            return response

        for trajectory_name, trajectory in [
            ("home_to_pick", request.plan.home_to_pick),
            ("pick_to_home", request.plan.pick_to_home),
            ("home_to_place", request.plan.home_to_place),
            ("place_to_home", request.plan.place_to_home),
        ]:
            valid, message = self._validate_trajectory(trajectory_name, trajectory)
            if not valid:
                result.success = False
                result.failure_reason = ExecutionResult.REASON_MALFORMED_TRAJECTORY
                result.message = message
                response.result = result
                return response

        valid, message = self._validate_endpoint_target(
            name="home_to_pick",
            trajectory=request.plan.home_to_pick,
            target_pose=piece.pose,
            ee_frame=request.layout.ee_frame,
            target_label=f"piece {request.plan.piece_id}",
            tolerance=max(
                self.endpoint_position_tolerance,
                float(request.layout.attach_distance_threshold),
            ),
        )
        if not valid:
            result.success = False
            result.failure_reason = ExecutionResult.REASON_MALFORMED_TRAJECTORY
            result.message = message
            response.result = result
            return response

        valid, message = self._validate_endpoint_target(
            name="home_to_place",
            trajectory=request.plan.home_to_place,
            target_pose=request.layout.cell_poses[int(request.plan.cell_id)],
            ee_frame=request.layout.ee_frame,
            target_label=f"cell {request.plan.cell_id}",
            tolerance=max(
                self.endpoint_position_tolerance,
                float(request.layout.attach_distance_threshold),
            ),
        )
        if not valid:
            result.success = False
            result.failure_reason = ExecutionResult.REASON_MALFORMED_TRAJECTORY
            result.message = message
            response.result = result
            return response

        result.success = True
        result.failure_reason = ExecutionResult.REASON_NONE
        result.message = "Turn plan accepted for simulation replay."
        response.result = result
        return response

    def _validate_trajectory(
        self, name: str, trajectory: RobotTrajectory
    ) -> tuple[bool, str]:
        joint_trajectory = trajectory.joint_trajectory
        if list(joint_trajectory.joint_names) != EXPECTED_JOINTS:
            return False, f"{name} joint names must match the Panda 7-DOF arm."
        if len(joint_trajectory.points) == 0:
            return False, f"{name} must contain at least one trajectory point."
        for point in joint_trajectory.points:
            if len(point.positions) != len(EXPECTED_JOINTS):
                return (
                    False,
                    f"{name} contains a point with the wrong number of joint positions.",
                )
        return True, ""

    def _validate_endpoint_target(
        self,
        name: str,
        trajectory: RobotTrajectory,
        target_pose,
        ee_frame: str,
        target_label: str,
        tolerance: float,
    ) -> tuple[bool, str]:
        final_positions = trajectory.joint_trajectory.points[-1].positions
        tcp_position = self._tcp_position(
            joint_names=trajectory.joint_trajectory.joint_names,
            positions=final_positions,
            ee_frame=ee_frame,
        )
        target_position = np.array(
            [
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
            ],
            dtype=float,
        )
        error = float(np.linalg.norm(tcp_position - target_position))
        if error > tolerance:
            return (
                False,
                f"{name} endpoint misses {target_label} by {error:.3f} m (tolerance {tolerance:.3f} m).",
            )
        return True, ""

    def _tcp_position(
        self, joint_names: list[str], positions, ee_frame: str
    ) -> np.ndarray:
        cfg = {name: float(value) for name, value in zip(joint_names, positions)}
        cfg.update(FINGER_JOINTS)
        self.urdf.update_cfg(cfg)
        try:
            return np.array(self.urdf.get_transform(ee_frame)[:3, 3], dtype=float)
        except ValueError:
            if ee_frame != "panda_hand_tcp":
                raise
            base_to_link8 = self.urdf.get_transform("panda_link8")
            link8_to_tcp = np.eye(4, dtype=float)
            link8_to_tcp[2, 3] = LINK8_TCP_Z_OFFSET
            base_to_tcp = base_to_link8 @ link8_to_tcp
            return np.array(base_to_tcp[:3, 3], dtype=float)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TicTacToeRefereeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
