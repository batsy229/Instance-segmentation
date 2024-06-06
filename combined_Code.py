import cv2
import numpy as np
import math
import rospy
import geometry_msgs.msg
from msg_package.msg import action_msg
from realsense_depth import DepthCamera
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pyrealsense2 as rs
import time

model = YOLO("home/goku/Downloads/best_(1).pt").to("cuda")
names = model.model.names
dc = DepthCamera()
top_face_class_name = "top_face"

def transform_coordinates(x, y, z):
    cosAngle = math.cos(math.radians(-145))
    sinAngle = math.sin(math.radians(-145))
    translation = np.array([[-165], [188], [-73]])
    yT = y * cosAngle - z * sinAngle
    zT = y * sinAngle + z * cosAngle
    pointT = np.array([[x], [yT], [zT]])
    pointF = pointT + translation
    return pointF.flatten()

def publish_coordinates(x, y, z):
    rospy.init_node('d435_node', anonymous=True)
    pub = rospy.Publisher('/ur3pose', action_msg, queue_size=1)
    pose_target = geometry_msgs.msg.Pose()
    pose_target.position.x = x
    pose_target.position.y = y
    pose_target.position.z = z
    pose_target.orientation.y = 1.0
    pose_target.orientation.w = 0
    time.sleep(5)
    pub.publish(pose_target)

try:
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        if not ret:
            continue
        results = model.predict(source=color_frame, conf=0.70)
        annotator = Annotator(color_frame, line_width=2)
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            confs = results[0].boxes.conf.cpu().tolist()
            for mask, box, cls, conf in zip(masks, boxes, clss, confs):
                color = colors(int(cls), True)
                color = tuple(map(int, color))
                color_frame[mask] = color_frame[mask] * 0.5 + np.array(color) * 0.5
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cz = depth_frame[cy, cx]
                label = f"{names[int(cls)]}: {conf:.2f} (X:{cx}, Y:{cy}, Z:{cz}mm)"
                annotator.box_label(box, label, color=color)
                if names[int(cls)] == top_face_class_name:
                    xT, yT, zT = transform_coordinates(cx, cy, cz)
                    publish_coordinates(xT, yT, zT)
        cv2.imshow('RealSense with YOLO', color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    dc.release()
    cv2.destroyAllWindows()
