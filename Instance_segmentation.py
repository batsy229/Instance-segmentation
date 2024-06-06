import cv2
import numpy as np
from realsense_depth_new import DepthCamera  
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pyrealsense2 as rs

model = YOLO("C:\\Users\\gokul\\Downloads\\best.pt").to("cuda")
names = model.model.names

dc = DepthCamera()

top_face_class_name = "top_face"  

try:
    while True:
        ret, depth_frame, color_frame = dc.get_frame()

        if not ret:
            continue  

        
        results = model.predict(source=color_frame, conf=0.60)

        # Annotate the image
        annotator = Annotator(color_frame, line_width=2)
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            confs = results[0].boxes.conf.cpu().tolist()  

            for mask, box, cls, conf in zip(masks, boxes, clss, confs):
                mask = mask.astype(bool)
                color = colors(int(cls), True)
                color = tuple(map(int, color))  
                color_frame[mask] = color_frame[mask] * 0.5 + np.array(color) * 0.5

                if names[int(cls)] == top_face_class_name:
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    cz = depth_frame[cy, cx]

                    # Draw bounding box, label with confidence score, and centroid
                    label = f"{names[int(cls)]}: {conf:.2f}"
                    annotator.box_label(box, label, color=color)
                    cv2.circle(color_frame, (cx, cy), radius=2, color=color, thickness=-1)
                    depth_point = rs.rs2_deproject_pixel_to_point(dc.intr, [cx, cy], cz)
                    print(f"Center: (X: {cx}, Y: {cy}, Z: {cz}mm)")
                    print('point coordinates', depth_point)
                    
                else:
                    # Draw bounding box and label for other classes
                    annotator.box_label(box, f"{names[int(cls)]} {conf:.2f}", color=color)
                    cv2.rectangle(color_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        print('intrinsics values', dc.intr)
        depth_frame_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_frame_normalized = np.uint8(depth_frame_normalized)
        depth_colormap = cv2.applyColorMap(depth_frame_normalized, cv2.COLORMAP_JET)
        cv2.imshow('RealSense with YOLO', color_frame)
        cv2.imshow('depth with YOLO', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    dc.release()
    cv2.destroyAllWindows()