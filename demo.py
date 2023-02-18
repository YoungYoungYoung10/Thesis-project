"""
! author: Young
! date: 2022-9-18
主要功能：检测孩子是否在看电视，看了多久，距离多远
使用技术点：人脸检测、人脸识别（采集照片、训练、识别）、姿态估计
"""
import cv2,time
from pose_estimator import PoseEstimator
import numpy as np
import dlib
from utils import Utils
from argparse import ArgumentParser
# import asyncio

import websocket    # pip install websocket-client
import threading

ws = websocket.WebSocketApp("ws://localhost:12345/")
wst = threading.Thread(target=ws.run_forever)
wst.daemon = True
wst.start()


class MonitorBabay:
    def __init__(self):
        
        # 人脸68个关键点
        self.landmark_predictor = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")
        # 人脸检测
        self.face_detector = dlib.get_frontal_face_detector()
        # 站在1.5M远处，左眼最左边距离右眼最右边的像素距离(请使用getEyePixelDist方法校准，然后修改这里的值)
        self.eyeBaseDistance = 65
        # pose_estimator.show_3d_model()
        self.utils = Utils()

    
    # 获取两个眼角像素距离
    def getEyePixelDist(self):
        
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 姿态估计
        self.pose_estimator = PoseEstimator(img_size=(height, width))
        
        fpsTime = time.time()

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
           
            pixel_dist = 0

            for face in faces:
                
                # 关键点
                landmarks = self.landmark_predictor(gray, face)
                image_points = self.pose_estimator.get_image_points(landmarks)

                left_x = int(image_points[36][0])
                left_y = int(image_points[36][1])
                right_x = int(image_points[45][0])
                right_y = int(image_points[45][1])

                pixel_dist = abs(right_x-left_x)

                cv2.circle(frame, (left_x, left_y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (right_x, right_y), 8, (255, 0, 255), -1)

                # 人脸框
                frame = self.utils.draw_face_box(face,frame,'','','')
              

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            person_distance = round(self.pose_estimator.get_distance(self.eyeBaseDistance),2)
            if person_distance > 0.5:
                ws.send(str(person_distance))

            # if person_distance > 10:
            #     websocket_send('play')
            # else:
            #     websocket_sned('pause')
            frame = self.utils.cv2AddText(frame, "FPS: " + str(int(fps_text)),  (20, 30), textColor=(0, 255, 0), textSize=50)
            frame = self.utils.cv2AddText(frame,  "Distance: " + str(person_distance) +"m", (20, 100), textColor=(0, 255, 0), textSize=50)
           
            # frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Distance Measuring', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

    # 运行主程序

m = MonitorBabay()


# 参数
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='distance')
                    
parser.add_argument("--display", type=int, default=1,
                    help="mode of display")     
                     
parser.add_argument("--w", type=int, default=960,
                    help="frame width")   
parser.add_argument("--h", type=int, default=720,
                    help="frame height")                           
args = parser.parse_args()


mode = args.mode



if  mode == 'distance':
    m.getEyePixelDist()

