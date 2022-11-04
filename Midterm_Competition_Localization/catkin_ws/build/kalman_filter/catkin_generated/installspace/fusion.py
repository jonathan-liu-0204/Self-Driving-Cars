#!/usr/bin/env python3

from tkinter import Scale
import rospy
import math, os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
# KF coded by yourself
# from Kalman_filter import KalmanFilter

import numpy as np

class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
                
        # State transition error covariance
        # self.Q = np.eye(3)*0.0009
        self.Q = np.array([[0.08, 0.0, 0.0],
                           [0.0, 0.009, 0.0], 
                           [0.0, 0.0, 1.0]])

        # Measurement error
        #self.R = np.eye(2)*0.08
        self.R = np.array([[0.5, 0.00],
                           [0.00, 0.7]])

    def predict(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        # raise NotImplementedError

    def update(self, z):
        
        S = self.R + np.dot(np.dot(self.H, self.P), self.H.T)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(3)
        # self.P = np.dot(I - np.dot(K, self.H), self.P)
        self.P = (I - np.matmul(K, self.H)) * self.P
        # raise NotImplementedError


class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.KF = None
        self.step = 0   # Record update times

        self.last_odometry_position = np.zeros(2)
        self.last_odometry_angle = 0

        self.gt_list = []
        self.est_list = []

    def shutdown(self):
        print("shuting down fusion.py")

    def gpsCallback(self, data):
        self.step += 1
        # Get GPS data only for 2D (x, y)
        measurement = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)[:2,:2]

        # KF update
        if self.step == 1:
            self.init_KF(measurement[0], measurement[1], 0)
        else:
            self.KF.R = np.eye(2)*0.05
            self.KF.update(z = [measurement[0], measurement[1]])
        print(f"estimation: {self.KF.x}")

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = [data.pose.pose.position.x, data.pose.pose.position.y]
        odometry_covariance = np.array(data.pose.covariance).reshape(6, -1)[:2,:2]

        # Get euler angle from quaternion
        roll, pitch, yaw = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

        # Calculate odometry difference
        diff = [position[0] - self.last_odometry_position[0], position[1] - self.last_odometry_position[1]]
        diff_yaw = yaw - self.last_odometry_angle

        # KF predict
        if self.step == 1:
            self.init_KF(position[0], position[1], 0)
        else:
            self.KF.Q[:2, :2] = np.eye(2)
            self.KF.predict(u = [diff[0], diff[1], diff_yaw])
        print(f"estimation: {self.KF.x}")
        self.last_odometry_position = position
        self.last_odometry_angle = yaw

        quaternion = quaternion_from_euler(0, 0, yaw)

        # Publish odometry with covariance
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        predPose.pose.pose.position.x = self.KF.x[0]
        predPose.pose.pose.position.y = self.KF.x[1]
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        predPose.pose.covariance = [self.KF.P[0][0], self.KF.P[0][1],0,0,0,0,
                                    self.KF.P[1][0], self.KF.P[1][1],0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0 ]
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        gt_position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.gt_list.append(gt_position)
        if self.KF is not None:
            kf_position = self.KF.x[:2]
            self.est_list.append(kf_position)

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        gt_x, gt_y = zip(*self.gt_list)
        est_x, est_y = zip(*self.est_list)
        plt.plot(gt_x, gt_y, alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(est_x, est_y, alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("/home/jonathan/catkin_ws/src/kalman_filter/results"):
            os.mkdir("/home/jonathan/catkin_ws/src/kalman_filter/results")
        plt.savefig("/home/jonathan/catkin_ws/src/kalman_filter/results/result.png")
        plt.show()

    def init_KF(self, x, y, yaw):
        # Initialize the Kalman filter when the first data comes in
        self.KF = KalmanFilter(x = x, y = y, yaw = yaw)
        self.KF.A = np.identity(3)
        self.KF.B = np.identity(3)

if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()
