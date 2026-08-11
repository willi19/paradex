#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

def publish_trajectory(positions, duration=1.0):
    # 创建一个 ROS Publisher
    pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)
    
    # 等待 ROS 初始化
    rospy.sleep(1)

    # 创建 JointTrajectory 消息
    traj = JointTrajectory()
    traj.header.seq = 0
    traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = ''
    traj.joint_names = ['right_index_1_joint', 'right_little_1_joint', 'right_middle_1_joint', 
                        'right_ring_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']
    
    # 创建一个点
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = []
    point.accelerations = []
    point.effort = []
    point.time_from_start = rospy.Duration(duration)

    # 计算发布次数和时间间隔
    rate = rospy.Rate(500)  # 每秒发布 500 次
    end_time = rospy.get_time() + duration  # 计算结束时间

    # 循环发布消息，直到达到指定的持续时间
    while rospy.get_time() < end_time:
        traj.header.stamp = rospy.Time.now()  # 更新时间戳
        traj.points = [point]  # 清空之前的点并添加新点
        rospy.loginfo("Publishing trajectory: {}".format(positions))
        pub.publish(traj)
        rate.sleep()  # 等待下一个循环

if __name__ == '__main__':
    try:
        # 初始化 ROS 节点
        rospy.init_node('gripper_control_node')

        # 动作 1
        publish_trajectory([0, 0, 0, 0, 0, 0], duration=1)
        rospy.sleep(1)
        
        # 动作 2
        publish_trajectory([1.2, 0.5, 0.5, 0.5, 0.4, 0], duration=1)
        rospy.sleep(1)
        
        # 动作 3
        publish_trajectory([1.6, 1.6, 1.6, 1.6, 0.4, 0], duration=1)

    except rospy.ROSInterruptException:
        pass

