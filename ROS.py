#! /usr/bin/env python3
 
import sys
import copy
import rospy
import geometry_msgs.msg
from msg_package.msg import action_msg
 
pose_target = geometry_msgs.msg.Pose()
pose_target.orientation.y = 1.0
pose_target.orientation.w = 0
 
if __name__ == '__main__':
    
    try:
        rospy.init_node('d435_node', anonymous=True)
        pub = rospy.Publisher('/ur3pose', action_msg, queue_size=1)
        rospy.spin()
    except KeyboardInterrupt:
        pass