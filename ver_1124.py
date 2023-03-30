import cv2
import os
import mediapipe as mp
import time
import numpy as np
import sys
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QCoreApplication
from PyQt5 import uic
from collections import deque
from pygame import mixer
from os import path
from pydub import AudioSegment

from scipy.ndimage.filters import gaussian_filter1d


np.seterr(divide='ignore', invalid='ignore')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

queue = deque()
queue_la = deque()
queue_ra = deque()
queue_ll = deque()
queue_rl = deque()
queue_bo = deque()
total_body_queue = [queue_la, queue_ra, queue_ll, queue_rl, queue_bo]
face_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
left_arm = {11, 13, 15, 17, 19, 21}
right_arm = {12, 14, 16, 18, 20, 22}
# left_leg = {23, 25, 27, 29, 31}
# right_leg = {24, 26, 28, 30, 32}
left_leg = {23, 25, 27, 29, 31}
right_leg = {24, 26, 28, 30, 32}
body = {11, 12, 23, 24}
# 0 ~ 32 총 33개의 점 66개의 좌표값
# 0 ~ 10 은 얼굴점 제외 11 ~ 32 22개의 점 44개의 좌표값

prevTime = 0
prev_time = 0
# FPS_S = 12.7
# FPS_S = 12.4
# FPS_S = 13.5 #whatislove 13 tictok2 15
FPS_S = [13.5 , 15,15,15,15,15]
min_ = 100

sim_threshold = 0.15
max_val = 0.5 # 0.25 0.3 0.35
frame_length = [30, 20, 10]

user_name = ""
user_rank = []
total_user_rank = []
video_num = 0
video_path = ["C:/users/HCH/Desktop/vsaa/motion/practice.mp4"]
# video_path = ["D:/pythonProject2/tomB.mp4", "D:/pythonProject2/easy.mp4", "D:/pythonProject2/tomB2.mp4", "D:/졸작/tictok.mp4"]
video_width = 1248
video_height = 702

# 코사인거리 반환하는 함수
def findCosineSimilarity_1(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return np.sqrt(2 * (1 - (a / (np.sqrt(b) * np.sqrt(c)))))
    # return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# 좌표들을 하나의 배열로 만들어줌
def get_position(lmlist):
    keyp_list = []
    features = [0] * 44
    k = -2
    for j in range(0, 22):
        k = k + 2
        try:
            if k >= 44:
                break
            features[k] = lmlist[j][0]
            features[k + 1] = lmlist[j][1]
        except:
            features[k] = 0
            features[k + 1] = 0
    keyp_list.append(features)

    return keyp_list

def adapt_center(center_body):
    center_x = 0
    center_y = 0

    for center in center_body:
        center_x = center_x + center[0]
        center_y = center_y + center[1]
    return center_x/4, center_y/4

# 필요한 좌표 값들을  list형태로 반환
def findPosition(img, landmarks):  # 이미지는 사람치 #find
    lmlist = []
    center_body = []
    h, w, c = img.shape  # 이미지의 높이, 너비, 채널(이미지의 층)을 추출한다.
    for id, lm in enumerate(landmarks):  # pose 의 landmark값을 열거합니다.
        if id not in body:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        center_body.append([cx, cy])
    center_x , center_y = adapt_center(center_body)

    for id, lm in enumerate(landmarks):  # pose 의 landmark값을 열거합니다.
        if id in face_num:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cx = (cx - center_x)
        cy = (cy - center_y) 
        lmlist.append([cx, cy])

    return lmlist

def cosine_each_body(num, web_body, cam_body):
    trans_body = Normalizer().fit([web_body])
    web_body = trans_body.transform([web_body])
    cam_body = trans_body.transform([cam_body])
    total_body_queue[num].append(web_body)

    if len(total_body_queue[num]) > 10:
        total_body_queue[num].popleft()  # 오래된 데이터 부터 삭제

    min_2 = 100
    for j in range(0, len(total_body_queue[num])):
        body_simS = findCosineSimilarity_1(total_body_queue[num][j][0], cam_body[0])
        if min_2 > body_simS:
            min_2 = body_simS
    min_2 = round(min_2, 5)
    return min_2


# 좌표들을 부위별 리스트에 저장후 반환
def cut_body(keyp_list):
    left_arm_list = []
    right_arm_list = []
    left_leg_list = []
    right_let_list = []
    body_list = []
    full_body = []
    for i in range(11, 33):
        spot = keyp_list[0][(i-11)*2], keyp_list[0][(i-11)*2+1]
        if i in body:
            body_list.append(spot[0])
            body_list.append(spot[1])
        if i in left_arm:
            left_arm_list.append(spot[0])
            left_arm_list.append(spot[1])
        elif i in right_arm:
            right_arm_list.append(spot[0])
            right_arm_list.append(spot[1])
        elif i in left_leg:
            left_leg_list.append(spot[0])
            left_leg_list.append(spot[1])
        elif i in right_leg:
            right_let_list.append(spot[0])
            right_let_list.append(spot[1])
    full_body.append(left_arm_list)
    full_body.append(right_arm_list)
    full_body.append(left_leg_list)
    full_body.append(right_let_list)
    full_body.append(body_list)
    return full_body

def sim_result_graph(result, part):
    i = 0
    while i < len(result):
        # if part == 'la' or part == 'ra':
        #     result[i] = result[i]*0.6
        # elif part == 'll' or part == 'rl' or part == 'bo':
        #     result[i] = result[i] + max_val*0.1
            
        if result[i] > max_val:
            result[i] = max_val
        result[i] = -result[i] + max_val
        result[i] = result[i] * (100 * 1/max_val)
        i+=1
        
    correct_line = np.zeros(len(result)) + 100 * (1 - sim_threshold/max_val)

    plt.clf()
    ax = plt.axes()
    
    result = gaussian_filter1d(result,sigma=1)
    ax.plot(result)
    ax.plot(correct_line, 'g')
    ax.xaxis.set_major_locator(plt.MultipleLocator(len(result) / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    plt.ylim([0, 100])
    plt.title('pose accuracy over time')
    plt.xlabel('video timeline')
    plt.ylabel('pose accuracy [%]')

    if part == 'all':
        plt.savefig('./sim_result_all.png')
    elif part == 'la':
        plt.savefig('./sim_result_la.png')
    elif part == 'ra':
        plt.savefig('./sim_result_ra.png')
    elif part == 'll':
        plt.savefig('./sim_result_ll.png')
    elif part == 'rl':
        plt.savefig('./sim_result_rl.png')
    elif part == 'bo':
        plt.savefig('./sim_result_bd.png')
    else:
        print('cannot find sim result')



Main_UI = uic.loadUiType("C:/users/HCH/Desktop/vsaa/motion/sample.ui")[0]
form_class = uic.loadUiType("C:/users/HCH/Desktop/vsaa/motion/graph.ui")[0]

class Main_page(QMainWindow, Main_UI):
    def __init__(self):
        super(Main_page, self).__init__()
        self.setupUi(self)
        self.set_video_name()
        self.start_btn.clicked.connect(self.click_start)
        self.start_btn.clicked.connect(QCoreApplication.instance().quit)
        self.video_list.currentIndexChanged.connect(self.combo_fun)
        self.open_btn.clicked.connect(self.click_open)
        self.name_edit.returnPressed.connect(self.enter_start)
        self.name_edit.returnPressed.connect(QCoreApplication.instance().quit)

    def set_video_name(self):
        global video_path
        for video in video_path:
            print(video)
            index = video.rfind('/') + 1
            last_index = video.rfind('.')
            self.video_list.addItem(video[index:])
            total_user_rank.append([])


    def click_start(self):
        global user_name
        if self.name_edit.text() == '':
            user_name = "None"
        else:
            user_name = self.name_edit.text()


    def enter_start(self):
        global user_name
        user_name = self.name_edit.text()


    def combo_fun(self):
        global video_num
        video_num = self.video_list.currentIndex()
        self.write_rank(total_user_rank[video_num])


    def write_rank(self, users):
        self.rank_list.clear()
        self.rank_list2.clear()
        self.rank_num.clear()
        count = 1
        for i in users:
            self.rank_list.addItem(str(i[1]))
            self.rank_list2.addItem(str(i[0])+" %")
            self.rank_num.addItem(str(count))
            count += 1

    def click_open(self):
        global video_path
        fname = QFileDialog.getOpenFileName(self,"File Load", './', 'Video(*.mp4 *.avi)')
        if fname[0]:
            index = fname[0].rfind('/') + 1
            self.video_list.addItem(fname[0][index:])
            video_path.append(fname[0])
            total_user_rank.append([])
            self.video_list.setCurrentIndex(self.video_list.count()-1)


    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, "end_check", "Do you want exit?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if ans == QMessageBox.Yes:
            self.close()
            global e
            e = None
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

class Graph(QMainWindow, form_class) :
    def __init__(self) :
        super(Graph, self).__init__()
        self.setupUi(self)

        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_all.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result)))

        self.pushButtonAL.clicked.connect(self.btnALfunction)
        self.pushButtonLA.clicked.connect(self.btnLAfunction)
        self.pushButtonRA.clicked.connect(self.btnRAfunction)
        self.pushButtonLL.clicked.connect(self.btnLLfunction)
        self.pushButtonRL.clicked.connect(self.btnRLfunction)
        self.pushButtonBO.clicked.connect(self.btnBOfunction)
        self.retry_btn.clicked.connect(self.click_retry)
        self.retry_btn.clicked.connect(QCoreApplication.instance().quit)

    def btnALfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_all.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result)))
        self.title.setText("Total Result")

    def btnLAfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_la.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result_ra)))
        self.title.setText("Left Arm Result")

    def btnRAfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_ra.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result_la)))
        self.title.setText("Right Arm Result")

    def btnLLfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_ll.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result_rl)))
        self.title.setText("Left Leg Result")

    def btnRLfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_rl.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result_ll)))
        self.title.setText("Right leg Result")

    def btnBOfunction(self):
        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("sim_result_bd.png")
        self.graphImg.setPixmap(self.qPixmapFileVar)
        self.score.setText("{:.0f}%".format(np.mean(sim_result_bo)))
        self.title.setText("Body Result")

    def click_retry(self):
        total_user_rank[video_num].append([round(np.mean(sim_result),0) , user_name])
        total_user_rank[video_num].sort(reverse = True)
        global e
        e.write_rank(total_user_rank[video_num])
        e.name_edit.setText("")

    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, "end_check", "Do you want exit?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if ans == QMessageBox.Yes:
            self.close()
            global e
            e = None
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

app = QApplication(sys.argv)
e = Main_page()
Result_Window = None

    # For webcam input:

while e is not None:
    sim_result = []
    sim_result_la = []
    sim_result_ra = []
    sim_result_ll = []
    sim_result_rl = []
    sim_result_bo = []

    queue.clear()
    queue_la.clear()
    queue_ra.clear()
    queue_ll.clear()
    queue_rl.clear()
    queue_bo.clear()
    total_body_queue = [queue_la, queue_ra, queue_ll, queue_rl, queue_bo]

    if Result_Window is not None:
        Result_Window.hide()
    if not e.isVisible():
        e.show()
        app.exec_()

    time.sleep(3)

    if e is None: break
    e.hide()
    filePath = video_path[video_num]
    src = filePath
    dst = src+".wav"
            # convert wav to mp3                                                            
    sound = AudioSegment.from_file(src)
    sound.export(dst, format="wav")
    
    webcap = cv2.VideoCapture(2)
    # webcap = cv2.VideoCapture("tomB.mp4")
    cap = cv2.VideoCapture(filePath)
    # player = MediaPlayer(filePath)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            model_complexity=0,
            min_tracking_confidence=0.5) as pose,\
        mp_pose.Pose(
            min_detection_confidence=0.5,
            model_complexity=0,
            min_tracking_confidence=0.5) as pose2:
        while cap.isOpened():
            success, image = cap.read()
            camsuccess, camimage = webcap.read()
            camimage = cv2.flip(camimage,1)
            # audio_frame, val = player.get_frame()
            if not success or not camsuccess:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            image.flags.writeable = False
            image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            camimage.flags.writeable = False
            camimage = cv2.resize(camimage, (720, 480), interpolation=cv2.INTER_AREA)
            camimage = cv2.cvtColor(camimage, cv2.COLOR_BGR2RGB)

            current_time = time.time() - prev_time
            if (success is True) and (current_time > 1. / FPS_S[video_num]):
                prev_time = time.time() # 시간을 저장하기 위함
                results = pose.process(image)
                camresults = pose2.process(camimage)
                if not results.pose_landmarks:
                    continue

                if camresults.pose_landmarks is None:
                    min_= 1
                    sim_result.append(min_)
                    sim_result_la.append(min_)
                    sim_result_ra.append(min_)
                    sim_result_ll.append(min_)
                    sim_result_rl.append(min_)
                    sim_result_bo.append(min_)
                    continue

                landmarks = results.pose_landmarks.landmark
                lmlist = findPosition(image, landmarks)

                camlandmarks = camresults.pose_landmarks.landmark
                camlmlist = findPosition(camimage, camlandmarks)

                # keyplist -> 신체 부위의 모든 좌표(얼굴 제외)를 가진 list  /  나머지는 부위별 좌표를 가진 list
                keyp_list = get_position(lmlist)
                # 0 -> left_arm / 1 -> right_arm / 2 -> left_leg / 3 -> right_leg / 4 -> body
                full_body = cut_body(keyp_list)

                cam_keyp_list = get_position(camlmlist)
                cam_full_body = cut_body(cam_keyp_list)
                
                body_sim_result = []
                for i in range(0, 5):
                    each_result = cosine_each_body(i, full_body[i], cam_full_body[i])
                    body_sim_result.append(each_result)

                sim_result_la.append(body_sim_result[0])
                sim_result_ra.append(body_sim_result[1])
                sim_result_ll.append(body_sim_result[2])
                sim_result_rl.append(body_sim_result[3])
                sim_result_bo.append(body_sim_result[4])

                #Normalizer
                transformer = Normalizer().fit(keyp_list)
                keyp_list = transformer.transform(keyp_list)
                cam_keyp_list = transformer.transform(cam_keyp_list)

                queue.append(keyp_list)
                if len(queue) > 5:
                    queue.popleft() # 오래된 데이터 부터 삭제
                min_ = 100

                ## 최근 5프레임 정도의 자세와 비교하는 코드
                for i in range(0, len(queue)):
                    simscore = findCosineSimilarity_1(queue[i][0], cam_keyp_list[0])
                    if min_ > simscore:
                        min_ = simscore

                min_ = round(min_, 5)
                sim_result.append(min_)
                # if min_ > sim_threshold:
                #     print(full_body[2])
                #     print(cam_full_body[2])
                #     print("\n")

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            camimage.flags.writeable = True
            camimage = cv2.cvtColor(camimage, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                camimage,
                camresults.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.resize(image, (video_width, video_height), interpolation=cv2.INTER_AREA)
            camimage =  cv2.resize(camimage, (480, 270), interpolation=cv2.INTER_AREA)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if min_ < sim_threshold:
                # cv2.putText(image, "CORRECT STEPS", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(image,(0,0),(video_width,video_height),(0,255,0),10)
            else:
                # cv2.putText(image, "NOT CORRECT STEPS", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(image, (0, 0), (video_width, video_height), (0, 0, 255), 10)

            if(len(sim_result) == 1):
                mixer.init()
                mixer.music.load(dst)
                mixer.music.play()
            # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            cv2.imshow('Video', image)
            cv2.moveWindow('Video', 0, 100)
            cv2.imshow('webcam', camimage)
            cv2.moveWindow('webcam', video_width+80, 100)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    max_index = len(sim_result)
    video_time = round(video_length/video_fps)

    # player.close_player()
    mixer.music.stop()
    cap.release()
    webcap.release()
    cv2.destroyAllWindows()


    def format_func(value, tick_number):
        if value == 0:
            return "0:00"
        elif value == max_index/4:
            sec_1 = round(video_time / 4) % 60
            min_1 = int(round(video_time / 4) / 60 % 60)
            return "{}:{:0>2}".format(min_1, sec_1)
        elif value == max_index*2/4:
            sec_2 = round(video_time * 2 / 4) % 60
            min_2 = int(round(video_time * 2 / 4) / 60 % 60)
            return "{}:{:0>2}".format(min_2, sec_2)
        elif value == max_index*3/4:
            sec_3 = round(video_time * 3 / 4) % 60
            min_3 = int(round(video_time * 3 / 4) / 60 % 60)
            return "{}:{:0>2}".format(min_3, sec_3)
        else:
            sec_4 = round(video_time) % 60
            min_4 = int(round(video_time) / 60 % 60)
            return "{}:{:0>2}".format(min_4, sec_4)

    sim_result_graph(sim_result, 'all')
    sim_result_graph(sim_result_la, 'ra')
    sim_result_graph(sim_result_ra, 'la')
    sim_result_graph(sim_result_ll, 'rl')
    sim_result_graph(sim_result_rl, 'll')
    sim_result_graph(sim_result_bo, 'bo')

    Result_Window = Graph()
    Result_Window.show()
    app.exec_()