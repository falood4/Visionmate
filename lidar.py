import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import time
from collections import deque


class DynamicLidarNavigator(Node):

    def __init__(self):
        super().__init__('dynamic_lidar_nav')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Distance thresholds (Hysteresis)
        self.ENTER_DANGER_DIST = 1.5
        self.EXIT_DANGER_DIST = 1.8

        # Motion detection
        self.APPROACH_SPEED = 0.6
        self.velocity_window = deque(maxlen=5)

        # Direction smoothing
        self.direction_window = deque(maxlen=10)

        # Memory
        self.prev_clusters = []
        self.prev_time = None

        self.obstacle_active = False
        self.last_direction_state = None
        self.last_distance = None
        self.last_alert_time = 0

        self.DIST_CHANGE_THRESHOLD = 0.7
        self.MIN_TIME_BETWEEN_ALERTS = 3.0
        self.REMINDER_INTERVAL = 6.0


    # --------------------------------------------------
    def cluster_obstacles(self, ranges):
        clusters = []
        cluster = []

        for i in range(len(ranges)):
            if i == 0:
                cluster.append(i)
            else:
                if abs(ranges[i] - ranges[i - 1]) < 0.3:
                    cluster.append(i)
                else:
                    clusters.append(cluster)
                    cluster = [i]

        if cluster:
            clusters.append(cluster)

        return clusters


    # --------------------------------------------------
    def scan_callback(self, msg):

        now = time.time()

        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
        angles_deg = np.degrees(angles)

        front_mask = np.logical_and(angles_deg > -90, angles_deg < 90)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        valid = np.logical_and(np.isfinite(front_ranges), front_ranges > 0.05)
        front_ranges = front_ranges[valid]
        front_angles = front_angles[valid]

        if len(front_ranges) == 0:
            return

        # Stable distance using median of closest 5 points
        min_distance = np.median(np.sort(front_ranges)[:5])

        # Hysteresis
        if not self.obstacle_active:
            danger = min_distance < self.ENTER_DANGER_DIST
        else:
            danger = min_distance < self.EXIT_DANGER_DIST

        if not danger:
            self.obstacle_active = False
            self.last_direction_state = None
            self.last_distance = None
            return

        # Detect direction
        best_index = np.argmin(front_ranges)
        raw_angle = np.degrees(front_angles[best_index])

        self.direction_window.append(raw_angle)
        smoothed_angle = np.mean(self.direction_window)

        # Categorize into stable states
        if smoothed_angle > 30:
            direction_state = "LEFT"
            alert_message = "Obstacle ahead. Move Left."
        elif smoothed_angle < -30:
            direction_state = "RIGHT"
            alert_message = "Obstacle ahead. Move Right."
        else:
            direction_state = "STRAIGHT"
            alert_message = "Obstacle ahead. Go Straight carefully."

        # Intelligent Trigger Logic
        current_time = time.time()
        trigger = False

        if not self.obstacle_active:
            trigger = True

        elif self.last_direction_state != direction_state:
            trigger = True

        elif (self.last_distance is not None and
              abs(min_distance - self.last_distance) > self.DIST_CHANGE_THRESHOLD):
            trigger = True

        elif current_time - self.last_alert_time > self.REMINDER_INTERVAL:
            trigger = True

        if trigger and current_time - self.last_alert_time > self.MIN_TIME_BETWEEN_ALERTS:

            self.get_logger().warn(alert_message)

            self.last_direction_state = direction_state
            self.last_distance = min_distance
            self.last_alert_time = current_time
            self.obstacle_active = True


def main():
    rclpy.init()
    node = DynamicLidarNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()