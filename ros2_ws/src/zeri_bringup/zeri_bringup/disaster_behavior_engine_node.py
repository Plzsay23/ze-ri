#!/usr/bin/env python3

import json
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

import tf2_ros


@dataclass
class PersonTarget:
    marker_id: int
    x: float
    y: float
    z: float
    last_seen: float


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class DisasterBehaviorEngineNode(Node):
    """
    Small mission behavior layer for Ze-Ri.

    Commands on /zeri/behavior/command:
      - go_person 1
      - go_nearest
      - auto_on
      - auto_off
      - cancel

    State is published as short text on /zeri/behavior/state.
    """

    def __init__(self):
        super().__init__("disaster_behavior_engine_node")

        self.declare_parameter("marker_topic", "/zeri/person_markers")
        self.declare_parameter("command_topic", "/zeri/behavior/command")
        self.declare_parameter("state_topic", "/zeri/behavior/state")
        self.declare_parameter("vlm_event_topic", "/zeri/behavior/vlm_event")
        self.declare_parameter("mission_event_topic", "/zeri/mission/event")
        self.declare_parameter("nav_action_name", "/navigate_to_pose")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("approach_distance_m", 0.30)
        self.declare_parameter("goal_z", 0.0)
        self.declare_parameter("auto_approach", False)
        self.declare_parameter("auto_revisit", False)
        self.declare_parameter("min_goal_interval_sec", 3.0)
        self.declare_parameter("person_fresh_timeout_sec", 5.0)
        self.declare_parameter("arrival_mode", "vlm_wait")
        self.declare_parameter("arrival_dwell_sec", 5.0)

        self.declare_parameter("enable_search", True)
        self.declare_parameter("search_waypoints", "")
        self.declare_parameter("search_step_m", 0.5)
        self.declare_parameter("search_pause_sec", 2.0)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.vlm_event_topic = str(self.get_parameter("vlm_event_topic").value)
        self.mission_event_topic = str(self.get_parameter("mission_event_topic").value)
        self.nav_action_name = str(self.get_parameter("nav_action_name").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.approach_distance_m = float(self.get_parameter("approach_distance_m").value)
        self.goal_z = float(self.get_parameter("goal_z").value)
        self.auto_approach = bool(self.get_parameter("auto_approach").value)
        self.auto_revisit = bool(self.get_parameter("auto_revisit").value)
        self.min_goal_interval_sec = float(self.get_parameter("min_goal_interval_sec").value)
        self.person_fresh_timeout_sec = float(self.get_parameter("person_fresh_timeout_sec").value)
        self.arrival_mode = str(self.get_parameter("arrival_mode").value)
        self.arrival_dwell_sec = float(self.get_parameter("arrival_dwell_sec").value)
        
        self.enable_search = bool(self.get_parameter("enable_search").value)
        self.search_waypoints_str = str(self.get_parameter("search_waypoints").value)
        self.search_step_m = float(self.get_parameter("search_step_m").value)
        self.search_pause_sec = float(self.get_parameter("search_pause_sec").value)

        self.search_waypoints = []
        self.search_waypoints_are_relative = not self.search_waypoints_str.strip()
        if self.search_waypoints_str.strip():
            for pt in self.search_waypoints_str.split(";"):
                if pt.strip():
                    try:
                        x_str, y_str = pt.split(",")
                        self.search_waypoints.append((float(x_str.strip()), float(y_str.strip())))
                    except ValueError:
                        pass
        if not self.search_waypoints:
            s = self.search_step_m
            self.search_waypoints = [
                (s, 0.0), (s, s), (0.0, s), (-s, s),
                (-s, 0.0), (-s, -s), (0.0, -s), (s, -s)
            ]
        self.search_wp_index = 0
        self.search_origin_xy = None

        self.people = {}
        self.visited = set()
        self.active_goal_handle = None
        self.active_goal_type = None
        self.active_person_id = None
        self.pending_person_after_cancel = None
        self.handoff_person_id = None
        self.handoff_until = 0.0
        self.last_goal_time = 0.0
        self.last_state = ""

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(MarkerArray, self.marker_topic, self.on_markers, marker_qos)
        self.create_subscription(String, self.command_topic, self.on_command, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.vlm_event_pub = self.create_publisher(String, self.vlm_event_topic, 10)
        self.mission_event_pub = self.create_publisher(String, self.mission_event_topic, 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        self.timer = self.create_timer(0.5, self.on_timer)

        self.publish_state("READY idle")
        self.get_logger().info(
            "disaster behavior engine started: "
            f"markers={self.marker_topic}, command={self.command_topic}, "
            f"state={self.state_topic}, mission_event={self.mission_event_topic}, "
            f"nav={self.nav_action_name}, "
            f"auto_approach={self.auto_approach}, arrival_mode={self.arrival_mode}"
        )

    def publish_state(self, text: str, force: bool = False):
        if force or text != self.last_state:
            msg = String()
            msg.data = text
            self.state_pub.publish(msg)
            self.get_logger().info(text)
            self.last_state = text

    def on_markers(self, msg: MarkerArray):
        now = time.time()
        for marker in msg.markers:
            if marker.ns != "fixed_person_positions":
                continue
            self.people[int(marker.id)] = PersonTarget(
                marker_id=int(marker.id),
                x=float(marker.pose.position.x),
                y=float(marker.pose.position.y),
                z=float(marker.pose.position.z),
                last_seen=now,
            )

    def on_command(self, msg: String):
        command = msg.data.strip()
        parts = command.split()
        if not parts:
            return

        op = parts[0].lower()

        if op == "go_person":
            if len(parts) < 2:
                self.publish_state("ERROR go_person needs id", force=True)
                return
            try:
                person_id = int(parts[1])
            except ValueError:
                self.publish_state(f"ERROR bad person id: {parts[1]}", force=True)
                return
            self.send_person_goal(person_id, explicit=True)
            return

        if op == "go_nearest":
            target = self.find_nearest_person(include_visited=True)
            if target is None:
                self.publish_state("NO_PERSON no marker target", force=True)
                return
            self.send_person_goal(target.marker_id, explicit=True)
            return

        if op == "auto_on":
            self.auto_approach = True
            self.publish_state("AUTO_ON waiting for person marker", force=True)
            return

        if op == "auto_off":
            self.auto_approach = False
            self.publish_state("AUTO_OFF idle", force=True)
            return

        if op == "cancel":
            self.cancel_active_goal()
            return

        if op == "reset_visited":
            self.visited.clear()
            self.publish_state("VISITED_RESET", force=True)
            return

        if op in ("resume", "complete_handoff"):
            self.finish_handoff("RESUME")
            return

        self.publish_state(f"ERROR unknown command: {command}", force=True)

    def on_timer(self):
        if self.active_goal_handle is not None:
            if self.active_goal_type == "search" and self.auto_approach:
                target = self.find_nearest_person(include_visited=self.auto_revisit)
                if target is not None:
                    self.pending_person_after_cancel = int(target.marker_id)
                    self.publish_state(f"SEARCH_INTERRUPT person={target.marker_id}", force=True)
                    self.cancel_active_goal()
            return

        if self.handoff_person_id is not None:
            if self.arrival_mode == "test_dwell" and time.time() >= self.handoff_until:
                self.finish_handoff("DWELL_DONE")
            else:
                remain = max(0.0, self.handoff_until - time.time())
                self.publish_state(
                    f"HANDOFF_WAIT person={self.handoff_person_id} "
                    f"mode={self.arrival_mode} remain={remain:.1f}"
                )
            return

        if self.auto_approach:
            target = self.find_nearest_person(include_visited=self.auto_revisit)
            if target is not None:
                self.send_person_goal(target.marker_id, explicit=False)
                return

            if time.time() - self.last_goal_time < self.min_goal_interval_sec:
                return

            if self.enable_search:
                if time.time() - self.last_goal_time >= self.search_pause_sec:
                    self.send_search_goal()
                return

        if not self.auto_approach:
            self.publish_state(f"IDLE people={len(self.people)} visited={len(self.visited)}")
            return

        self.publish_state(f"AUTO_WAIT people={len(self.people)} visited={len(self.visited)}")

    def send_search_goal(self):
        if not self.search_waypoints:
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.publish_state(f"WAIT_NAV action server missing: {self.nav_action_name}", force=True)
            return

        x, y = self.search_waypoints[self.search_wp_index]
        if self.search_waypoints_are_relative:
            if self.search_origin_xy is None:
                self.search_origin_xy = self.lookup_robot_xy()
                if self.search_origin_xy is None:
                    return
            x += self.search_origin_xy[0]
            y += self.search_origin_xy[1]
        
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = self.goal_z
        pose.pose.orientation.w = 1.0
        
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = ""
        
        self.active_goal_type = "search"
        self.last_goal_time = time.time()
        
        self.publish_state(f"AUTO_SEARCH_GOAL index={self.search_wp_index} goal=({x:.2f},{y:.2f})", force=True)
        
        future = self.nav_client.send_goal_async(goal, feedback_callback=self.on_nav_feedback)
        future.add_done_callback(self.on_goal_response)

    def find_nearest_person(self, include_visited: bool):
        robot = self.lookup_robot_xy()
        if robot is None:
            return None

        rx, ry = robot
        candidates = []
        now = time.time()
        for target in self.people.values():
            if not include_visited and target.marker_id in self.visited:
                continue
            if now - target.last_seen > self.person_fresh_timeout_sec:
                continue
            d = math.hypot(target.x - rx, target.y - ry)
            candidates.append((d, target))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def lookup_robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as exc:
            self.publish_state(f"WAIT_TF {self.target_frame}->{self.base_frame}: {exc}")
            return None

        t = tf.transform.translation
        return float(t.x), float(t.y)

    def make_approach_pose(self, target: PersonTarget):
        robot = self.lookup_robot_xy()
        if robot is None:
            return None

        rx, ry = robot
        vx = rx - target.x
        vy = ry - target.y
        dist = math.hypot(vx, vy)

        if dist < 1.0e-3:
            vx, vy = -1.0, 0.0
            dist = 1.0

        ux = vx / dist
        uy = vy / dist

        goal_x = target.x + ux * self.approach_distance_m
        goal_y = target.y + uy * self.approach_distance_m
        yaw = math.atan2(target.y - goal_y, target.x - goal_x)
        qx, qy, qz, qw = yaw_to_quat(yaw)

        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = self.goal_z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def send_person_goal(self, person_id: int, explicit: bool):
        if person_id not in self.people:
            self.publish_state(f"NO_PERSON id={person_id}", force=True)
            return

        if self.active_goal_handle is not None:
            self.publish_state("BUSY already navigating", force=True)
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.publish_state(f"WAIT_NAV action server missing: {self.nav_action_name}", force=True)
            return

        target = self.people[person_id]
        pose = self.make_approach_pose(target)
        if pose is None:
            return

        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = ""

        self.active_person_id = person_id
        self.active_goal_type = "person"
        self.last_goal_time = time.time()
        mode = "MANUAL" if explicit else "AUTO"
        self.publish_state(
            f"{mode}_GO_PERSON id={person_id} "
            f"goal=({pose.pose.position.x:.2f},{pose.pose.position.y:.2f})",
            force=True,
        )

        future = self.nav_client.send_goal_async(goal, feedback_callback=self.on_nav_feedback)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self.active_goal_type == "person":
                person_id = self.active_person_id
                self.active_person_id = None
                self.publish_state(f"NAV_REJECTED person={person_id}", force=True)
            else:
                self.publish_state(f"SEARCH_REJECTED index={self.search_wp_index}", force=True)
                self.search_wp_index = (self.search_wp_index + 1) % len(self.search_waypoints)
            self.active_goal_type = None
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_nav_result)
        if self.active_goal_type == "person":
            self.publish_state(f"NAV_ACCEPTED person={self.active_person_id}", force=True)
        else:
            self.publish_state(f"SEARCH_ACCEPTED index={self.search_wp_index}", force=True)

    def on_nav_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        remaining = float(feedback.distance_remaining)
        if self.active_goal_type == "person":
            self.publish_state(f"NAV_ACTIVE person={self.active_person_id} remaining={remaining:.2f}")
        else:
            self.publish_state(f"SEARCH_ACTIVE index={self.search_wp_index} remaining={remaining:.2f}")

    def on_nav_result(self, future):
        result = future.result()
        status = int(result.status)
        goal_type = self.active_goal_type

        self.active_goal_handle = None
        self.active_goal_type = None

        if goal_type == "person":
            person_id = self.active_person_id
            self.active_person_id = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.begin_arrival_handoff(person_id)
            elif status == GoalStatus.STATUS_CANCELED:
                self.publish_state(f"CANCELED person={person_id}", force=True)
            else:
                self.publish_state(f"FAILED person={person_id} status={status}", force=True)
        elif goal_type == "search":
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.publish_state(f"SEARCH_REACHED index={self.search_wp_index}", force=True)
                self.search_wp_index = (self.search_wp_index + 1) % len(self.search_waypoints)
                self.last_goal_time = time.time()
            elif status == GoalStatus.STATUS_CANCELED:
                self.publish_state("SEARCH_CANCELED", force=True)
                pending_person_id = self.pending_person_after_cancel
                self.pending_person_after_cancel = None
                if pending_person_id is not None:
                    self.send_person_goal(int(pending_person_id), explicit=False)
            else:
                self.publish_state(f"SEARCH_FAILED index={self.search_wp_index} status={status}", force=True)
                self.search_wp_index = (self.search_wp_index + 1) % len(self.search_waypoints)
                self.last_goal_time = time.time()

    def begin_arrival_handoff(self, person_id):
        if person_id is None:
            self.publish_state("ARRIVED person=None", force=True)
            return

        person_id = int(person_id)
        self.handoff_person_id = person_id
        if self.arrival_mode == "test_dwell":
            self.handoff_until = time.time() + max(self.arrival_dwell_sec, 0.0)
        else:
            self.handoff_until = 0.0

        target = self.people.get(person_id)
        distance_m = None
        robot_xy = self.lookup_robot_xy()
        if target is not None and robot_xy is not None:
            distance_m = math.hypot(target.x - robot_xy[0], target.y - robot_xy[1])

        payload = {
            "event": "arrived_at_person",
            "person_id": person_id,
            "selected_person_id": f"person_{person_id}",
            "arrival_mode": self.arrival_mode,
            "test_dwell_sec": self.arrival_dwell_sec,
            "target_frame": self.target_frame,
            "behavior_command_topic": self.command_topic,
            "source": "disaster_behavior_engine_node",
            "stamp_sec": time.time(),
            "target_context": {
                "distance_m": distance_m,
                "source": "nav2_behavior_engine",
                "arrival_reason": "nav2_goal_succeeded",
                "person_marker_id": person_id,
                "target_frame": self.target_frame,
            },
        }
        if target is not None:
            payload.update({
                "person_x": target.x,
                "person_y": target.y,
                "person_z": target.z,
            })

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.vlm_event_pub.publish(msg)
        self.mission_event_pub.publish(msg)

        self.publish_state(
            f"ARRIVED person={person_id} -> VLM_EVENT/MISSION_EVENT mode={self.arrival_mode}",
            force=True,
        )

        if self.arrival_mode not in ("test_dwell", "vlm_wait"):
            self.finish_handoff("ARRIVAL_DONE")

    def finish_handoff(self, reason):
        if self.handoff_person_id is None:
            self.publish_state("HANDOFF idle", force=True)
            return

        person_id = int(self.handoff_person_id)
        self.visited.add(person_id)
        self.handoff_person_id = None
        self.handoff_until = 0.0
        self.last_goal_time = time.time()
        self.publish_state(f"HANDOFF_DONE person={person_id} reason={reason}", force=True)

    def cancel_active_goal(self):
        if self.handoff_person_id is not None:
            person_id = self.handoff_person_id
            self.handoff_person_id = None
            self.handoff_until = 0.0
            self.publish_state(f"CANCELED_HANDOFF person={person_id}", force=True)
            return

        if self.active_goal_handle is None:
            self.publish_state("CANCEL idle", force=True)
            self.pending_person_after_cancel = None
            return

        if self.active_goal_type == "person":
            self.publish_state(f"CANCELING person={self.active_person_id}", force=True)
        else:
            self.publish_state(f"CANCELING search index={self.search_wp_index}", force=True)

        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(lambda _: self.publish_state("CANCEL_SENT", force=True))


def main(args=None):
    rclpy.init(args=args)
    node = DisasterBehaviorEngineNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
