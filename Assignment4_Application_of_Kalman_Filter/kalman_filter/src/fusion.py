#!/usr/bin/env python

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
from Kalman_filter import KalmanFilter

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
            self.KF.R = ???
            self.KF.update(z = ???)
        print(f"estimation: {self.KF.x}")

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = ???
        odometry_covariance = np.array(data.pose.covariance).reshape(6, -1)[:2,:2]

        # Get euler angle from quaternion
        roll, pitch, yaw = euler_from_quaternion(???)

        # Calculate odometry difference
        diff = ???
        diff_yaw = ???

        # KF predict
        if self.step == 1:
            self.init_KF(position[0], position[1], 0)
        else:
            self.KF.R = ???
            self.KF.predict(u = ???)
        print(f"estimation: {self.KF.x}")
        self.last_odometry_position = position
        self.last_odometry_angle = yaw

        quaternion = quaternion_from_euler(0, 0, ???)

        # Publish odometry with covariance
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        predPose.pose.pose.position.x = ???
        predPose.pose.pose.position.y = ???
        predPose.pose.pose.orientation.x = ???
        predPose.pose.pose.orientation.y = ???
        predPose.pose.pose.orientation.z = ???
        predPose.pose.pose.orientation.w = ???
        predPose.pose.covariance = [self.KF.P[0][0], self.KF.P[0][1],0,0,0,0,
                                    self.KF.P[1][0], self.KF.P[1][1],0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0 ]
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        gt_position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        kf_position = self.KF.x[:2]
        self.gt_list.append(gt_position)
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
        if not os.path.exists("/home/ee904/SDC/hw4/src/kalman_filter/results"):
            os.mkdir("/home/ee904/SDC/hw4/src/kalman_filter/results")
        plt.savefig("/home/ee904/SDC/hw4/src/kalman_filter/results/result.png")
        plt.show()

    def init_KF(self, x, y, yaw):
        # Initialize the Kalman filter when the first data comes in
        self.KF = KalmanFilter(x = x, y = y, yaw = yaw)
        self.KF.A = ???
        self.KF.B = ???

if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    # fusion.plot_path()
