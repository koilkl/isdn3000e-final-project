from __future__ import annotations

import argparse
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState as MoveItRobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState
from robot_descriptions.loaders.yourdfpy import load_robot_description

from ttt_layout import load_layout


PANDA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
HOME_POSITIONS = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)
FINGER_JOINTS = {
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

LINK8_QUAT_XYZW = (0.9238795325112867, -0.3826834323650898, 0.0, 0.0)
LINK8_TCP_Z_OFFSET = 0.1034

POSITION_TOLERANCE_M = 5e-4
ORIENTATION_TOLERANCE_RAD = 5e-3
MIN_SINGULAR_VALUE_THRESHOLD = 0.015
MAX_CONDITION_NUMBER = 250.0
JOINT_LIMIT_TOLERANCE = 1e-6
DEFAULT_VIEWER_PORT = 8081
MOVEIT_WARMUP_SEC = 5.0


@dataclass(frozen=True)
class AuditTarget:
    name: str
    kind: str
    tcp_position: tuple[float, float, float]
    source_id: int


@dataclass(frozen=True)
class AuditResult:
    target: AuditTarget
    joint_positions: tuple[float, ...]
    desired_tcp_position: tuple[float, float, float]
    solved_tcp_position: tuple[float, float, float]
    solved_tcp_wxyz: tuple[float, float, float, float]
    position_error_m: float
    orientation_error_rad: float
    min_singular_value: float
    condition_number: float


def build_link8_pose_from_tcp_target(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z + LINK8_TCP_Z_OFFSET)
    pose.orientation.x = LINK8_QUAT_XYZW[0]
    pose.orientation.y = LINK8_QUAT_XYZW[1]
    pose.orientation.z = LINK8_QUAT_XYZW[2]
    pose.orientation.w = LINK8_QUAT_XYZW[3]
    return pose


def build_audit_targets() -> list[AuditTarget]:
    layout = load_layout().workspace
    targets: list[AuditTarget] = []
    for piece in layout.initial_pieces:
        targets.append(
            AuditTarget(
                name=f"piece_{piece.piece_id}",
                kind="piece",
                tcp_position=(
                    float(piece.pose.position.x),
                    float(piece.pose.position.y),
                    float(piece.pose.position.z),
                ),
                source_id=int(piece.piece_id),
            )
        )
    for cell_id, pose in enumerate(layout.cell_poses):
        targets.append(
            AuditTarget(
                name=f"cell_{cell_id}",
                kind="cell",
                tcp_position=(
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ),
                source_id=cell_id,
            )
        )
    return targets


def load_panda_urdf():
    return load_robot_description(
        "panda_description",
        load_meshes=True,
        build_scene_graph=True,
        load_collision_meshes=False,
    )


def _arm_cfg_map(joint_positions: tuple[float, ...] | list[float]) -> dict[str, float]:
    cfg = {name: float(value) for name, value in zip(PANDA_JOINT_NAMES, joint_positions)}
    cfg.update(FINGER_JOINTS)
    return cfg


def get_joint_limits(robot) -> dict[str, tuple[float, float]]:
    return {
        name: (
            float(robot.joint_map[name].limit.lower),
            float(robot.joint_map[name].limit.upper),
        )
        for name in PANDA_JOINT_NAMES
    }


def compute_tcp_transform(robot, joint_positions: tuple[float, ...] | list[float]) -> np.ndarray:
    robot.update_cfg(_arm_cfg_map(joint_positions))
    return np.array(robot.get_transform("panda_hand_tcp"), dtype=float)


def compute_link8_transform(robot, joint_positions: tuple[float, ...] | list[float]) -> np.ndarray:
    robot.update_cfg(_arm_cfg_map(joint_positions))
    return np.array(robot.get_transform("panda_link8"), dtype=float)


def extract_wxyz_from_transform(transform: np.ndarray) -> tuple[float, float, float, float]:
    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        )

    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    if index == 0:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return (
            (rotation[2, 1] - rotation[1, 2]) / s,
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
        )
    if index == 1:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return (
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
        )
    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return (
        (rotation[1, 0] - rotation[0, 1]) / s,
        (rotation[0, 2] + rotation[2, 0]) / s,
        (rotation[1, 2] + rotation[2, 1]) / s,
        0.25 * s,
    )


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    angle = math.acos(cosine)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-9:
        return 0.5 * skew
    return (angle / (2.0 * math.sin(angle))) * skew


def compute_orientation_error(actual_transform: np.ndarray, desired_pose: Pose) -> float:
    desired_rotation = xyzw_to_rotation_matrix(
        desired_pose.orientation.x,
        desired_pose.orientation.y,
        desired_pose.orientation.z,
        desired_pose.orientation.w,
    )
    delta = actual_transform[:3, :3] @ desired_rotation.T
    return float(np.linalg.norm(rotation_log(delta)))


def xyzw_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def compute_jacobian(robot, joint_positions: tuple[float, ...] | list[float], delta: float = 1e-5) -> np.ndarray:
    base = np.array(joint_positions, dtype=float)
    base_transform = compute_tcp_transform(robot, base)
    base_rotation = base_transform[:3, :3]
    base_position = base_transform[:3, 3]
    jacobian = np.zeros((6, len(base)), dtype=float)
    for index in range(len(base)):
        perturbed = base.copy()
        perturbed[index] += delta
        perturbed_transform = compute_tcp_transform(robot, perturbed)
        jacobian[:3, index] = (perturbed_transform[:3, 3] - base_position) / delta
        delta_rotation = perturbed_transform[:3, :3] @ base_rotation.T
        jacobian[3:, index] = rotation_log(delta_rotation) / delta
    return jacobian


def compute_singularity_metrics(robot, joint_positions: tuple[float, ...] | list[float]) -> tuple[float, float]:
    singular_values = np.linalg.svd(compute_jacobian(robot, joint_positions), compute_uv=False)
    min_singular_value = float(singular_values[-1])
    condition_number = float(singular_values[0] / max(min_singular_value, 1e-12))
    return min_singular_value, condition_number


def assert_joint_limits(joint_positions: tuple[float, ...] | list[float], joint_limits: dict[str, tuple[float, float]]) -> None:
    for name, value in zip(PANDA_JOINT_NAMES, joint_positions):
        lower, upper = joint_limits[name]
        if value < lower - JOINT_LIMIT_TOLERANCE or value > upper + JOINT_LIMIT_TOLERANCE:
            raise AssertionError(
                f"{name}={value:.6f} outside joint limits [{lower:.6f}, {upper:.6f}]"
            )


class IKAuditClient(Node):
    def __init__(self) -> None:
        super().__init__("ik_pose_audit")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

    def wait_for_service(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.ik_client.wait_for_service(timeout_sec=0.5):
                time.sleep(MOVEIT_WARMUP_SEC)
                return True
        return False

    def compute_ik(
        self,
        target_pose: Pose,
        seed_positions: tuple[float, ...] | list[float] = HOME_POSITIONS,
        timeout_sec: float = 10.0,
    ) -> tuple[float, ...]:
        seed_state = JointState()
        seed_state.name = list(PANDA_JOINT_NAMES)
        seed_state.position = [float(value) for value in seed_positions]

        request = GetPositionIK.Request()
        request.ik_request.group_name = "panda_arm"
        request.ik_request.robot_state = MoveItRobotState(joint_state=seed_state)
        request.ik_request.pose_stamped.header.frame_id = "panda_link0"
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.timeout.sec = 5
        request.ik_request.avoid_collisions = False

        future = self.ik_client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            raise TimeoutError("Timed out waiting for /compute_ik response.")

        response = future.result()
        if response is None or response.error_code.val != 1:
            raise RuntimeError(
                f"IK failed with code {getattr(getattr(response, 'error_code', None), 'val', 'N/A')}"
            )

        joint_positions = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        missing = [name for name in PANDA_JOINT_NAMES if name not in joint_positions]
        if missing:
            raise RuntimeError(f"IK solution missing joints: {missing}")
        return tuple(float(joint_positions[name]) for name in PANDA_JOINT_NAMES)


def audit_target(
    client: IKAuditClient,
    robot,
    joint_limits: dict[str, tuple[float, float]],
    target: AuditTarget,
) -> AuditResult:
    target_pose = build_link8_pose_from_tcp_target(*target.tcp_position)
    joint_positions = client.compute_ik(target_pose)
    assert_joint_limits(joint_positions, joint_limits)

    tcp_transform = compute_tcp_transform(robot, joint_positions)
    solved_tcp_position = tuple(float(value) for value in tcp_transform[:3, 3])
    position_error_m = float(np.linalg.norm(tcp_transform[:3, 3] - np.array(target.tcp_position)))
    orientation_error_rad = compute_orientation_error(
        compute_link8_transform(robot, joint_positions),
        target_pose,
    )
    min_singular_value, condition_number = compute_singularity_metrics(robot, joint_positions)

    if position_error_m > POSITION_TOLERANCE_M:
        raise AssertionError(
            f"{target.name} TCP position error {position_error_m:.6f} m exceeds {POSITION_TOLERANCE_M:.6f} m"
        )
    if orientation_error_rad > ORIENTATION_TOLERANCE_RAD:
        raise AssertionError(
            f"{target.name} orientation error {orientation_error_rad:.6f} rad exceeds {ORIENTATION_TOLERANCE_RAD:.6f} rad"
        )
    if min_singular_value < MIN_SINGULAR_VALUE_THRESHOLD:
        raise AssertionError(
            f"{target.name} minimum singular value {min_singular_value:.6f} below threshold {MIN_SINGULAR_VALUE_THRESHOLD:.6f}"
        )
    if condition_number > MAX_CONDITION_NUMBER:
        raise AssertionError(
            f"{target.name} condition number {condition_number:.3f} exceeds threshold {MAX_CONDITION_NUMBER:.3f}"
        )

    return AuditResult(
        target=target,
        joint_positions=joint_positions,
        desired_tcp_position=target.tcp_position,
        solved_tcp_position=solved_tcp_position,
        solved_tcp_wxyz=extract_wxyz_from_transform(tcp_transform),
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        min_singular_value=min_singular_value,
        condition_number=condition_number,
    )


def launch_moveit_stack(workspace_root: Path) -> subprocess.Popen[str]:
    command = (
        "source /opt/ros/humble/setup.bash && "
        f"cd {workspace_root} && "
        "if [ -f install/setup.bash ]; then source install/setup.bash; fi && "
        "ros2 launch ttt_bringup moveit_ik.launch.py"
    )
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=workspace_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_audit() -> list[AuditResult]:
    robot = load_panda_urdf()
    joint_limits = get_joint_limits(robot)
    targets = build_audit_targets()
    client = IKAuditClient()
    try:
        if not client.wait_for_service():
            raise RuntimeError("/compute_ik did not become available.")
        return [audit_target(client, robot, joint_limits, target) for target in targets]
    finally:
        client.destroy_node()


def add_workspace_geometry(server, targets: list[AuditTarget]) -> None:
    from ttt_layout import load_layout as _load_layout

    layout = _load_layout().workspace
    server.scene.add_grid("/grid", width=1.0, height=1.0)
    for target in targets:
        color = (80, 200, 80) if target.kind == "piece" else (220, 60, 60)
        z_size = float(layout.piece_size.z) if target.kind == "piece" else 0.005
        server.scene.add_box(
            name=f"/targets/{target.name}/object",
            position=target.tcp_position,
            dimensions=(float(layout.piece_size.x), float(layout.piece_size.y), z_size),
            color=color,
            opacity=0.7,
        )



def populate_viser_scene(results: list[AuditResult], host: str, port: int) -> None:
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(host=host, port=port, label="IK Pose Audit")
    desired_wxyz = (
        LINK8_QUAT_XYZW[3],
        LINK8_QUAT_XYZW[0],
        LINK8_QUAT_XYZW[1],
        LINK8_QUAT_XYZW[2],
    )
    server.gui.add_markdown(
        "\n".join(
            [
                "# IK Pose Audit",
                f"Loaded `{len(results)}` solved targets.",
                "Green boxes are stock pieces. Red boxes are board cells.",
                "Each target gets its own Panda instance plus desired/solved TCP frames.",
            ]
        )
    )

    add_workspace_geometry(server, [result.target for result in results])
    for result in results:
        robot = load_panda_urdf()
        root = f"/robots/{result.target.name}"
        viser_urdf = ViserUrdf(server, urdf_or_path=robot, root_node_name=root, load_meshes=True)
        viser_urdf.update_cfg(_arm_cfg_map(result.joint_positions))
        server.scene.add_frame(
            name=f"/frames/{result.target.name}/desired_tcp",
            position=result.desired_tcp_position,
            wxyz=desired_wxyz,
            axes_length=0.035,
            axes_radius=0.0025,
            origin_radius=0.004,
        )
        server.scene.add_frame(
            name=f"/frames/{result.target.name}/solved_tcp",
            position=result.solved_tcp_position,
            wxyz=result.solved_tcp_wxyz,
            axes_length=0.05,
            axes_radius=0.004,
            origin_radius=0.006,
        )
        server.scene.add_label(
            name=f"/frames/{result.target.name}/metrics",
            text=(
                f"{result.target.name}\\n"
                f"err={result.position_error_m * 1000.0:.2f} mm\\n"
                f"sigma_min={result.min_singular_value:.3f}"
            ),
            position=(
                result.solved_tcp_position[0],
                result.solved_tcp_position[1],
                result.solved_tcp_position[2] + 0.12,
            ),
        )

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MoveIt IK reachability for all board cells and stock pieces.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--once", action="store_true", help="Run the audit once, print a summary, and exit without starting viser.")
    filtered_args = rclpy.utilities.remove_ros_args(args=None)[1:]
    return parser.parse_args(filtered_args)


def main() -> None:
    args = parse_args()
    rclpy.init()
    try:
        results = run_audit()
    finally:
        rclpy.shutdown()
    if args.once:
        for result in results:
            print(
                f"{result.target.name}: "
                f"err={result.position_error_m * 1000.0:.2f}mm "
                f"sigma_min={result.min_singular_value:.3f} "
                f"cond={result.condition_number:.2f}"
            )
        return
    populate_viser_scene(results, host=args.host, port=args.port)
