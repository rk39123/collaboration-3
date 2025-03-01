import numpy as np
np.float = float

import sys
import cv2
import sqlite3
import datetime
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTabWidget, QScrollArea
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import rclpy
from rclpy.node import Node

# 추가 import
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Odometry
import math
import tf_transformations

import os
import time
import matplotlib.pyplot as plt
from collections import Counter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class LandmineApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # 30ms마다 프레임 업데이트
        self.status_messages = []  # 상태 메시지를 저장할 리스트

        # rclpy 초기화
        rclpy.init()
        self.node = rclpy.create_node('robo_go_node')
        self.publisher = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # 드론 좌표를 받을 변수들
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0

        # 로봇의 현재 위치/방향을 받을 변수들
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # 드론 좌표 구독
        self.node.create_subscription(
            Point,
            '/drone_position',          # drone_pose_subscriber 노드가 퍼블리시하는 토픽
            self.drone_pose_callback,   # 콜백 함수
            10
        )

        # 로봇의 odom 구독
        self.node.create_subscription(
            Odometry,
            '/odom',                    # turtlebot3_gazebo가 퍼블리시하는 odometry
            self.odom_callback,
            10
        )

        # Drone 카메라 관련 subscription
        self.bridge = CvBridge()
        self.latest_image = None
        self.node.create_subscription(Image, '/simple_drone/bottom/image_raw', self.image_callback, 10)

    def initUI(self):
        self.setWindowTitle('Landmine Detection')
        self.setGeometry(100, 100, 1020, 600)

        # 탭 위젯 설정
        self.tabs = QTabWidget(self)
        
        # 'Admin' 탭 추가
        self.admin_tab = QWidget()
        self.initAdminTab()
        self.tabs.addTab(self.admin_tab, "Admin")

        # 'Graph' 탭 추가
        self.graph_tab = QWidget()
        self.initGraphTab()
        self.tabs.addTab(self.graph_tab, "Graph")

        # 메인 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def initAdminTab(self):
        # Admin 탭에 들어갈 내용 설정
        adminLayout = QHBoxLayout()

        # 카메라 화면을 표시할 QLabel
        self.videoLabel = QLabel(self)
        self.videoLabel.setFixedSize(600, 380)

        # 상태 정보 표시
        self.statusTitleLabel = QLabel("상태", self)
        self.statusTitleLabel.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.statusTitleLabel.setFixedHeight(30)
        self.statusLabel = QLabel("대기 중", self)
        self.statusLabel.setStyleSheet("background-color: white; border: 1px solid #999999")
        self.statusLabel.setFixedHeight(100)

        # 드론 좌표를 표시할 라벨
        self.dronePositionLabel = QLabel("드론 위치: (x=0.0, y=0.0, z=0.0)", self)

        # 버튼 설정
        self.startButton = QPushButton("제거 시작", self)
        self.startButton.setStyleSheet("background-color: #8B0000; color: white;")
        self.startButton.clicked.connect(self.startRemoval)

        self.returnButton = QPushButton("귀환", self)
        self.returnButton.setStyleSheet("background-color: #4682B4; color: white;")
        self.returnButton.clicked.connect(self.returnToBase)

        self.researchButton = QPushButton("재탐색", self)
        self.researchButton.clicked.connect(self.research)

        # 테이블 설정
        self.dbTable = QTableWidget(self)
        self.dbTable.setColumnCount(3)
        self.dbTable.setHorizontalHeaderLabels(["번호", "위치", "시간"])
        self.dbTable.cellClicked.connect(self.onCellClicked)

        scrollArea = QScrollArea(self)
        scrollArea.setWidget(self.dbTable)
        scrollArea.setWidgetResizable(True)

        # Admin 탭 레이아웃
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.videoLabel)
        leftLayout.addWidget(self.statusTitleLabel)
        leftLayout.addWidget(self.statusLabel)
        leftLayout.addWidget(self.dronePositionLabel)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.startButton)
        buttonLayout.addWidget(self.returnButton)
        buttonLayout.addWidget(self.researchButton)
        leftLayout.addLayout(buttonLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(scrollArea)

        adminLayout.addLayout(leftLayout)
        adminLayout.addLayout(rightLayout)

        self.admin_tab.setLayout(adminLayout)

        # 데이터베이스 로드
        self.loadDatabase()
        
    def onCellClicked(self, row, column):
        self.dbTable.selectRow(row)

    def initGraphTab(self):
        graphLayout = QVBoxLayout()
        self.graphLabel = QLabel(self)
        graphLayout.addWidget(self.graphLabel)
        self.updateGraph()
        self.graph_tab.setLayout(graphLayout)

    def updateGraph(self):
        conn = sqlite3.connect('landmine.db')
        cursor = conn.cursor()

        cursor.execute("SELECT mine_location FROM search")
        rows = cursor.fetchall()

        mine_locations = []
        for row in rows:
            location_str = row[0].strip()
            match = re.match(r"^(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)$", location_str)
            if match:
                x = float(match.group(1))
                y = float(match.group(3))
                mine_locations.append((x, y))

        if mine_locations:
            x_coords, y_coords = zip(*mine_locations)
        else:
            x_coords, y_coords = [], []

        from collections import Counter
        mine_frequency = Counter(mine_locations)

        def categorize(x, y):
            if x < -2.0:
                return 'Close'
            elif -2.0 <= x <= 0:
                return 'Medium'
            else:
                return 'Far'

        categories = [categorize(x, y) for x, y in mine_locations]
        category_mapping = {'Close': 0, 'Medium': 1, 'Far': 2}
        category_numbers = [category_mapping[c] for c in categories]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x_coords, y_coords, c=category_numbers, cmap='viridis', marker='x',
            s=[mine_frequency[(x, y)]*10 for x, y in mine_locations]
        )

        plt.title("Mine Locations with Frequency and Categories")
        plt.xlabel("X Coordinates")
        plt.ylabel("Y Coordinates")
        plt.grid(True)
        plt.legend(*scatter.legend_elements(), title="Categories")

        for (x, y), freq in mine_frequency.items():
            plt.text(x, y, str(freq), fontsize=9, ha='right')

        plt.savefig("/tmp/mine_locations_with_frequency_and_categories.png")    

        pixmap = QPixmap("/tmp/mine_locations_with_frequency_and_categories.png")
        self.graphLabel.setPixmap(pixmap)
        plt.close()

        conn.close()

    def updateStatusLabel(self):
        display_messages = "\n".join(self.status_messages[-3:])
        self.statusLabel.setText(display_messages)
        
    def image_callback(self, msg):
        self.latest_image = msg

    def drone_pose_callback(self, point_msg):
        self.drone_x = point_msg.x
        self.drone_y = point_msg.y
        self.drone_z = point_msg.z
        self.node.get_logger().info(
            f"Drone position updated: (x={self.drone_x:.2f}, y={self.drone_y:.2f}, z={self.drone_z:.2f})"
        )

    def odom_callback(self, msg):
        """
        /odom 콜백: 로봇의 현재 (x, y, yaw) 값을 갱신
        """
        pose = msg.pose.pose
        self.robot_x = pose.position.x
        self.robot_y = pose.position.y

        # 쿼터니언 -> 오일러 변환
        orientation_q = pose.orientation
        quat_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(quat_list)
        self.robot_yaw = yaw

    def updateFrame(self):
        """
        PyQt의 주기적 타이머로 돌며 GUI 업데이트 & rclpy.spin_once 처리
        """
        rclpy.spin_once(self.node, timeout_sec=0.001)
        self.dronePositionLabel.setText(
            f"드론 위치: (x={self.drone_x:.2f}, y={self.drone_y:.2f}, z={self.drone_z:.2f})"
        )

        if self.latest_image is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            except CvBridgeError as e:
                print("CvBridge Error:", e)
                return
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = cv_image.shape
            bytesPerLine = ch * w
            qImg = QImage(cv_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(600, 380, Qt.KeepAspectRatio)
            self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

    def startRemoval(self):
        """
        제거 시작 버튼: 드론의 현재 (drone_x, drone_y) 위치로
        Turtlebot을 이동시킴(장애물 고려 없이 직선 이동)
        """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 1) DB 저장용으로 현재 드론 좌표를 문자열로 생성
        current_location = f"{self.drone_x:.2f},{self.drone_y:.2f}"
        self.saveToDatabase(current_location, current_time)

        # 2) 상태 업데이트
        message = f"제거 출동 {current_location} time: {current_time}"
        self.status_messages.append(str(message))
        self.updateStatusLabel()

        # 3) 목표 위치를 드론의 현재 x, y로 설정
        target_x = self.drone_x
        target_y = self.drone_y

        # 4) 직선 이동 함수 호출 (장애물 무시)
        self.move_straight_to_target(target_x, target_y)

    def move_straight_to_target(self, target_x, target_y):
        """
        로봇의 현재 위치 (self.robot_x, self.robot_y)에서
        (target_x, target_y)까지 직선으로 이동(간단 P 제어).
        """
        self.node.get_logger().info(f"Starting direct move to ({target_x:.2f}, {target_y:.2f})")

        # 목표 지점까지의 거리가 일정 이하(예: 0.2m)면 도달했다고 판단
        goal_tolerance = 0.2

        while rclpy.ok():
            # 이벤트 루프를 돌면서 콜백 처리
            rclpy.spin_once(self.node, timeout_sec=0.1)

            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            dist = math.sqrt(dx*dx + dy*dy)

            # 1) 목표 지점에 충분히 가까우면 정지
            if dist < goal_tolerance:
                cmd = Twist()
                self.cmd_vel_publisher.publish(cmd)  # 정지
                self.node.get_logger().info("Arrived at target, stopping.")
                break

            # 2) 각도 및 거리 계산
            desired_yaw = math.atan2(dy, dx)
            yaw_error = desired_yaw - self.robot_yaw
            # -π ~ π 범위로 보정
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            # 간단한 P 제어
            angular_z = 0.5 * yaw_error
            # 회전 각도가 너무 크면 전진속도를 0으로
            linear_x = 0.2 if abs(yaw_error) < 0.2 else 0.0

            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.cmd_vel_publisher.publish(cmd)

            # 0.1초마다 제어
            time.sleep(0.1)

    def saveToDatabase(self, location, time):
        conn = sqlite3.connect('landmine.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO search (mine_location, search_time) VALUES (?, ?)", (location, time))
        conn.commit()
        conn.close()
        self.loadDatabase()

    def loadDatabase(self):
        conn = sqlite3.connect('landmine.db')
        cursor = conn.cursor()
        cursor.execute("SELECT no, mine_location, search_time FROM search")
        rows = cursor.fetchall()

        self.dbTable.setRowCount(len(rows))
        self.dbTable.setColumnWidth(0, 15)
        self.dbTable.setColumnWidth(1, 110)
        self.dbTable.setColumnWidth(2, 150)

        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                self.dbTable.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

        conn.close()

    def returnToBase(self):
        """
        Turtlebot을 (home_x, home_y) 위치로 귀환.
        (직선 이동 방식 그대로 사용 가능)
        """
        home_x = -0.968901
        home_y = 1.997310
        home_z = 0.0  # 지면 주행이므로 일반적으로 0.0

        message = f"Turtlebot 귀환 중: x={home_x}, y={home_y}"
        self.status_messages.append(message)
        self.updateStatusLabel()

        self.node.get_logger().info(f"Returning to base: ({home_x}, {home_y})")
        # 간단히 move_straight_to_target 이용
        self.move_straight_to_target(home_x, home_y)

    def research(self):
        row = self.dbTable.currentRow()
        if row != -1:
            location = self.dbTable.item(row, 1).text()
            land_no = self.dbTable.item(row, 0).text()
            message = f"재탐색  {land_no}번  {location}"
            self.status_messages.append(str(message))
            self.updateStatusLabel()

            x, y = location.split(",")
            x = float(x)
            y = float(y)
            self.moveRobotToLocation(x, y)

    def moveRobotToLocation(self, x, y):
        """
        기존 test.py에 있던 재탐색용 함수.
        여기서도 Nav2 대신 직접 이동 방식을 사용하려면
        self.move_straight_to_target(x, y)를 호출해도 됨.
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'odom'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0
        self.publisher.publish(goal_msg)

        move_cmd = Twist()
        move_cmd.linear.x = 0.1
        move_cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(move_cmd)


def main(args=None):
    app = QApplication(sys.argv)
    ex = LandmineApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
