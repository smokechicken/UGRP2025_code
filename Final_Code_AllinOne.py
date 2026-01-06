###


import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

import os
import math
import math as m
import numpy as np
from dynamixel_sdk import *  # Dynamixel 모터 제어 라이브러리
import time


# =========================
# 데이터 구조(반환용)
# =========================
@dataclass
class DetectedObject:
    cls_id: int
    cls_name: str
    x_mm: float
    y_mm: float
    conf: float
    cx_px: float
    cy_px: float


def undistort_if_available(frame, calib_npz_path="calib.npz"):
    if not os.path.exists(calib_npz_path):
        return frame, None

    data = np.load(calib_npz_path)
    camera_matrix = data["cameraMatrix"]
    dist_coeffs = data["distCoeffs"]

    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undist, (camera_matrix, dist_coeffs, new_camera_matrix)


# =========================
# 빨간 사각형(박스) 검출
# =========================
def find_red_rectangle_corners(frame_bgr, min_area=30000):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 30, 30])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 30, 30])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_area = area
            best = approx

    if best is None:
        return None, mask

    pts = best.reshape(4, 2).astype(np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # TL
    rect[2] = pts[np.argmax(s)]       # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # TR
    rect[3] = pts[np.argmax(diff)]    # BL (origin)

    return rect, mask


# =========================
# 호모그래피 계산/변환
# =========================
def compute_homography_img_to_mm(box_img_pts, width_mm=500.0, height_mm=300.0):
    dst_mm = np.array([
        [0.0,        height_mm],   # UL
        [width_mm,   height_mm],   # UR
        [width_mm,   0.0],         # LR
        [0.0,        0.0],         # LL (origin)
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(box_img_pts, dst_mm)


def img_point_to_mm(pt_uv, H):
    p = np.array([[[pt_uv[0], pt_uv[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0, 0]
    return float(out[0]), float(out[1])


def is_valid_mm(x_mm: float, y_mm: float, width_mm: float, height_mm: float) -> bool:
    if np.isnan(x_mm) or np.isnan(y_mm):
        return False
    margin = 10.0
    return (-margin <= x_mm <= width_mm + margin) and (-margin <= y_mm <= height_mm + margin)


# =========================
# 클러스터링(20x20mm) 유틸
# =========================
def is_close_20x20(a: DetectedObject, b: DetectedObject, th_mm: float) -> bool:
    return abs(a.x_mm - b.x_mm) <= th_mm and abs(a.y_mm - b.y_mm) <= th_mm


def cluster_and_average_by_class(
    detections: List[DetectedObject],
    th_mm: float = 20.0,
) -> List[DetectedObject]:
    """
    클래스(cls_id)별로 분리해서 20x20mm 이내 값들을 평균으로 병합.
    """
    # cls_id별로 분리
    by_cls: Dict[int, List[DetectedObject]] = {}
    for d in detections:
        by_cls.setdefault(d.cls_id, []).append(d)

    merged_all: List[DetectedObject] = []

    for cls_id, dets in by_cls.items():
        dets_sorted = sorted(dets, key=lambda d: d.conf, reverse=True)
        clusters: List[List[DetectedObject]] = []

        for d in dets_sorted:
            assigned = False
            for cluster in clusters:
                rep = cluster[0]
                if is_close_20x20(rep, d, th_mm=th_mm):
                    cluster.append(d)
                    assigned = True
                    break
            if not assigned:
                clusters.append([d])

        # cluster -> 평균으로 합치기
        for cluster in clusters:
            xs = [c.x_mm for c in cluster]
            ys = [c.y_mm for c in cluster]
            cs = [c.conf for c in cluster]
            cxs = [c.cx_px for c in cluster]
            cys = [c.cy_px for c in cluster]

            x_mean = float(np.mean(xs))
            y_mean = float(np.mean(ys))
            conf_max = float(np.max(cs))

            best = max(cluster, key=lambda t: t.conf)

            merged_all.append(
                DetectedObject(
                    cls_id=best.cls_id,
                    cls_name=best.cls_name,
                    x_mm=x_mean,
                    y_mm=y_mean,
                    conf=conf_max,
                    cx_px=float(np.mean(cxs)),
                    cy_px=float(np.mean(cys)),
                )
            )

    merged_all.sort(key=lambda d: d.conf, reverse=True)
    return merged_all


# =========================
# 핵심 함수 (멀티 클래스)
# =========================
def get_object_coordinates_for_robot(
    model_path: str,
    camera_index: int = 0,
    allowed_class_names: Optional[List[str]] = None,  # None이면 모든 클래스 허용
    width_mm: float = 500.0,
    height_mm: float = 500.0,
    conf_thres: float = 0.4,
    imgsz: int = 640,
    max_frames: int = 120,
    require_box: bool = True,
    calib_npz_path: str = "calib.npz",
    visualize: bool = False,
    merge_threshold_mm: float = 20.0,
) -> list[tuple[float, float, str, float]]:
    """
    반환:
      [(cls_name, x_mm, y_mm, conf), ...]  (conf 내림차순)
    """

    model = YOLO(model_path)

    # 허용 클래스 이름 -> 허용 cls_id set으로 변환
    allowed_ids: Optional[set] = None
    if allowed_class_names is not None:
        allowed_ids = set()
        for k, v in model.names.items():
            if v in allowed_class_names:
                allowed_ids.add(int(k))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    collected_all: List[DetectedObject] = []
    H_cached = None

    try:
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_u, _ = undistort_if_available(frame, calib_npz_path)
            vis = frame_u.copy()

            box_pts, mask = find_red_rectangle_corners(frame_u)
            if box_pts is not None:
                H = compute_homography_img_to_mm(box_pts, width_mm=width_mm, height_mm=height_mm)
                H_cached = H
            else:
                H = H_cached

            if require_box and H is None:
                if visualize:
                    cv2.putText(vis, "Red box not found", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("CV", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            results = model.predict(source=frame_u, imgsz=imgsz, conf=conf_thres, verbose=False)
            res = results[0]

            if res.boxes is None or len(res.boxes) == 0:
                if visualize:
                    cv2.putText(vis, "No detection", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("CV", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            # (A) 프레임 로컬
            frame_local: List[DetectedObject] = []

            for box, cls_t, conf_t in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                cls_id = int(cls_t.item())
                conf = float(conf_t.item())

                if allowed_ids is not None and cls_id not in allowed_ids:
                    continue

                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                cls_name = model.names.get(cls_id, str(cls_id))

                if H is not None:
                    x_mm, y_mm = img_point_to_mm((cx, cy), H)
                else:
                    x_mm, y_mm = float("nan"), float("nan")

                if not is_valid_mm(x_mm, y_mm, width_mm, height_mm):
                    continue

                frame_local.append(
                    DetectedObject(
                        cls_id=cls_id,
                        cls_name=cls_name,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        conf=conf,
                        cx_px=cx,
                        cy_px=cy
                    )
                )

                if visualize:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                    cv2.putText(vis, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(vis, f"({x_mm:.1f},{y_mm:.1f})mm", (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # (B) 프레임 내부 병합(클래스별)
            frame_local_merged = cluster_and_average_by_class(frame_local, th_mm=merge_threshold_mm)

            # (C) 누적
            collected_all.extend(frame_local_merged)

            if visualize:
                cv2.imshow("CV", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        cap.release()
        if visualize:
            cv2.destroyAllWindows()

    # (D) 전체 프레임 병합(클래스별)
    merged_final = cluster_and_average_by_class(collected_all, th_mm=merge_threshold_mm)

    # 반환 포맷: (name, x, y, conf)
    result = [(d.x_mm, d.y_mm, d.cls_name, d.conf) for d in merged_final]

    # 디버그 출력
    print("=== Final merged objects (class-aware clustering) ===")
    for i, d in enumerate(merged_final[:50]):
        print(f"{i:02d}: {d.cls_name:10s} conf={d.conf:.3f} (x,y)=({d.x_mm:.1f},{d.y_mm:.1f}) mm")
    print(f"Final unique objects: {len(result)}")

    print("???")
    return result


##################################################################
##################################################################

# 상수 설정
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
# ADDR_MX_PRESENT_POSITION = 3625

PROTOCOL_VERSION = 2.0

DXL_IDS = [1, 2, 3, 4, 5, 6]  # Dynamixel IDs for the joints
DXL_IDS_123 = [1, 2, 3]  # Dynamixel XM540-W150
DXL_IDS_456 = [4, 5, 6]  # Dynamixel XD540-T270

BAUDRATE = 57600
DEVICENAME = "/dev/cu.usbserial-FT89FLNY"

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_MOVING_STATUS_THRESHOLD = 10

DXL_MIN_POSITION_VALUE = 0
DXL_MAX_POSITION_VALUE = 1023

L1, L2, L3, L4, L5 = 103, 253, 103, 76, 118.481


# 회전 행렬을 생성하는 함수
def get_rotation_matrix(axis, theta):
    """
    주어진 축(axis)과 각도(theta)에 대해 회전 행렬을 반환합니다.
    axis: 회전할 축 ('x', 'y', 'z')
    theta: 회전 각도 (라디안)
    """
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, m.cos(theta), -m.sin(theta)],
                         [0, m.sin(theta), m.cos(theta)]])
    elif axis == 'y':
        return np.array([[m.cos(theta), 0, m.sin(theta)],
                         [0, 1, 0],
                         [-m.sin(theta), 0, m.cos(theta)]])
    elif axis == 'z':
        return np.array([[m.cos(theta), -m.sin(theta), 0],
                         [m.sin(theta), m.cos(theta), 0],
                         [0, 0, 1]])

# 포트 초기화 ( 포트 열고 보율 설정, 실패 시 프로그램 종료)
portHandler = PortHandler(DEVICENAME)  # Dynamixel 모터 통신 담당
packetHandler = PacketHandler(PROTOCOL_VERSION)  # Dynamixel 통신 프로토콜 처리

if not portHandler.openPort():
    print("Failed to open the port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()

# Dynamixel 토크 활성화 및 속도 설정(통신 및 오류 상태를 확인할 수 있는 코드)
for dxl_id in DXL_IDS:
    dxl_comm_result1, dxl_error1 = packetHandler.write1ByteTxRx(portHandler, dxl_id, 64,
                                                                TORQUE_ENABLE)  # Torque Enable ADDR
    if dxl_comm_result1 != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result1))
    elif dxl_error1 != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error1))
    dxl_comm_result2, dxl_error2 = packetHandler.write4ByteTxRx(portHandler, dxl_id, 112, 30)  # Profile Speed ADDR, VAL
    if dxl_comm_result2 != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result2))
    elif dxl_error2 != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error2))

for i in range(1, 7):
    # 속도 제한
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 112, 20)
    # 가속도 제한
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 108, 30)


# 역기구학 계산 함수
def calculate_ik(x, y, z, tar_ori):
    # En(n = 1~6) : angle of each axe
    settingangle0 = 180
    settingangle1 = 180
    settingangle2 = 90
    settingangle3 = 90
    settingangle4 = 270
    settingangle5 = 0

    # position which last three axes intersect: (X, Y, Z)
    X = x - L5 * tar_ori[0][0]
    Y = y - L5 * tar_ori[1][0]
    Z = z - L5 * tar_ori[2][0]
    E1 = m.atan2(Y, X)
    A = m.atan2(L4, L3)
    E3 = m.acos(
        (L2 ** 2 + L3 ** 2 + L4 ** 2 - X ** 2 - Y ** 2 - (Z - L1) ** 2) / (2 * L2 * ((L3 ** 2 + L4 ** 2) ** 0.5))) + A
    E3 = np.pi / 2 - E3
    k = m.acos((L2 ** 2 + X ** 2 + Y ** 2 + (Z - L1) ** 2 - L3 ** 2 - L4 ** 2) / (
                2 * L2 * (X ** 2 + Y ** 2 + (Z - L1) ** 2) ** 0.5))

    E2 = m.acos((L1 ** 2 + (Z - L1) ** 2 - Z ** 2) / (2 * L1 * (X ** 2 + Y ** 2 + (Z - L1) ** 2) ** 0.5)) + k
    E2 = np.pi - E2

    R10 = np.array([[m.cos(E1), -m.sin(E1), 0], [m.sin(E1), m.cos(E1), 0], [0, 0, 1]])
    R21 = np.array([[m.cos(E2), 0, m.sin(E2)], [0, 1, 0], [-m.sin(E2), 0, m.cos(E2)]])
    R32 = np.array([[m.cos(E3), 0, m.sin(E3)], [0, 1, 0], [-m.sin(E3), 0, m.cos(E3)]])
    R30 = R32 @ R21 @ R10
    R30_inv = np.linalg.inv(R30)
    R63 = tar_ori @ R30_inv

    E4 = m.atan2(R63[1][0], R63[0][0])
    E5 = m.asin(-R63[2][0])
    E6 = m.atan2(R63[2][1], R63[2][2])

    E1 = E1 * 180 / m.pi
    E1 = E1 + 180
    E2 = -E2 * 180 / m.pi
    E2 = E2 + 180 + 5
    E3 = -E3 * 180 / m.pi
    E3 = E3 + 90
    E4 = E4 * 180 / m.pi
    E4 = E4 + 90
    E5 = -E5 * 180 / m.pi
    E5 = E5 + 270
    E6 = E6 * 180 / m.pi
    E6 = E6 + 0

    E1 = E1 % 360
    E2 = E2 % 360
    E3 = E3 % 360
    E4 = E4 % 360
    E5 = E5 % 360
    E6 = E6 % 360

    E1 = E1 * m.pi / 180
    E2 = E2 * m.pi / 180
    E3 = E3 * m.pi / 180
    E4 = E4 * m.pi / 180
    E5 = E5 * m.pi / 180
    E6 = E6 * m.pi / 180

    print([E1 * 180 / m.pi, E2 * 180 / m.pi, E3 * 180 / m.pi, E4 * 180 / m.pi, E5 * 180 / m.pi, E6 * 180 / m.pi])
    return [E1, E2, E3, E4, E5, E6]


# 각도 -> 모터 위치로 변환
def radians_to_dxl_position_456(angle):
    position = (angle) / (2 * m.pi * 5 / 6) * 1024
    return position


def radians_to_dxl_position_123(angle):
    position = (angle) / (2 * m.pi) * 4096
    return position


# 로봇팔을 목표 위치로 이동시키는 함수
def move_robot_arm(goal_positions):
    for i, dxl_id in enumerate(DXL_IDS):  # 0,e1 / 1,e2 / 2,e3
        dxl_goal_position = radians_to_dxl_position_123(goal_positions[i])
        dxl_goal_position = int(dxl_goal_position)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, 116,
                                                                  dxl_goal_position)  # Goal Position ADDR
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))


## 이 아래 함수는 어떤 모터를 쓰느냐에 따라서 radians_to_dxl_position을 바꿔줘야한다.
def move_i_robot_arm(i, goal_position):
    dxl_goal_position = radians_to_dxl_position_123(goal_position)
    dxl_goal_position = int(dxl_goal_position)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 116, dxl_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))


'''
# 7번 모터 제어 함수
def move_motor_7(goal_position):
    dxl_goal_position = int(goal_position)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID_7, ADDR_MX_GOAL_POSITION, dxl_goal_position)
    if dxl_comm_result != COMM_SUCCESS:
        print("7번 모터: %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("7번 모터: %s" % packetHandler.getRxPacketError(dxl_error))
'''


# 메인 프로그램
def pick_the_cup(x_box, y_box, z_box):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, 5, 112, 40)
    for once in range(1):
        # 목표 좌표(x, y, z)를 입력받음
        x, y, z = L4 + L5, 0, L1 + L2 + L3

        tar_ori = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]
                            ])
        dxl_present_position_of_M0, dxl_comm_result_of_M0, dxl_error_of_M0 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[0], 132
        )
        dxl_present_position_of_M1, dxl_comm_result_of_M1, dxl_error_of_M1 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[1], 132
        )
        dxl_present_position_of_M2, dxl_comm_result_of_M2, dxl_error_of_M2 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[2], 132
        )
        dxl_present_position_of_M3, dxl_comm_result_of_M3, dxl_error_of_M3 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[3], 132
        )
        dxl_present_position_of_M4, dxl_comm_result_of_M4, dxl_error_of_M4 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[4], 132
        )
        dxl_present_position_of_M5, dxl_comm_result_of_M5, dxl_error_of_M5 = packetHandler.read4ByteTxRx(
            portHandler, DXL_IDS[5], 132
        )
        print(dxl_present_position_of_M0 * 360.0 / 4095.0, dxl_present_position_of_M1 * 360.0 / 4095.0,
              dxl_present_position_of_M2 * 360.0 / 4095.0, dxl_present_position_of_M3 * 360.0 / 4095.0,
              dxl_present_position_of_M4 * 360.0 / 4095.0, dxl_present_position_of_M5 * 360.0 / 4095.0)
        joint_angles = calculate_ik(x, y, z, tar_ori)
        move_robot_arm(joint_angles)
        time.sleep(5)

        while True:
            moving_flags = []

            for dxl_id in DXL_IDS:
                moving, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, dxl_id, 122)
                moving_flags.append(moving)

            if all(m == 0 for m in moving_flags):
                break  # 전부 도착 → 다음 동작으로
        # 역기구학을 사용하여 목표 위치와 자세에 해당하는 조인트 각도 계산

        ##초기의 로봇팔 모터들의 배열을 (a00, a01, a02, a03, a04, a05) 라고 하자
        Fth_int_angle = [3072, 1024]

        ##이때 z_int 각도를 조금 높게 한다.
        tar_ori_int = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        x_int = x
        y_int = y
        z_int = z
        tar_ori_int = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        int_angles = calculate_ik(x_int, y_int, z_int, tar_ori_int)
        # 최종 도달 각도인 (an0, an1, an2, an3n an4, an5,) 를 구하자

        z_hypo_box1 = z_box + 30
        ## z_hpyo_box 는 그릇의 최대 높이보다 높게
        z_angle = math.atan2(y_box, x_box)
        tar_ori_box = get_rotation_matrix('z', z_angle)
        hypo_box_angles = calculate_ik(x_box, y_box, z_hypo_box1, tar_ori_box)
        hypo_box_angles[0] = hypo_box_angles[0]
        move_robot_arm(hypo_box_angles)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, 5, 112, 20)
        time.sleep(3)
        time.sleep(7)

        # 로봇팔을 최종 목표 위치로 이동
        # 그리퍼의 수평을 유지하며 도달하게 하기 위해서 모터의 속도 조절
        for i in range(1, 7):
            # 속도 제한
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 112, 30)
        for i in range(10):
            zlist = []
            for j in range(1, 11):
                zlist.append(z_hypo_box1 - (z_hypo_box1 - z_box) * j / 10)
            box_angles = calculate_ik(x_box, y_box, zlist[i], tar_ori_box)
            move_robot_arm(box_angles)
            time.sleep(0.0025)
            if i == 9:
                time.sleep(3)
        time.sleep(7)
        #################
        ## 그리퍼 작동 프로그램
        #################

        for i in range(10):
            zlist = []
            for j in range(1, 11):
                zlist.append(z_box - (z_box - z_hypo_box1) * j / 10)
            #box_angles = calculate_ik(x_box -100/10*i, y_box, zlist[i], tar_ori_box)
            box_angles = calculate_ik(x_box, y_box, zlist[i], tar_ori_box)
            move_robot_arm(box_angles)
            time.sleep(0.0025)
            if i == 9:
                time.sleep(3)

        '''
        x_p = L3+L5
        y_p = 0
        z_p = L1+L2-L4

        for i in range(1, 11):
            box_angles = calculate_ik(x_box - (x_box - L3+L5)/10*i, y_box - y_box/10*i, z_box - (z_box-L1+L2-L4)/10*i, tar_ori_box)
            move_robot_arm(box_angles)
            time.sleep(0.0025)
            if i == 9:
                time.sleep(3)
        '''

        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, 1, 112, 15)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, 1, 116, Fth_int_angle[0])
        time.sleep(6)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, 1, 112, 20)

        for i in range(1, 7):
            # 속도 제한
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 112, 30)
        z_hypo_box2 = z_box + 30
        for i in range(10):
            zlist = []
            for j in range(1, 11):
                zlist.append(z_hypo_box2 - (z_hypo_box2 - z_box) * j / 10)
            box_angles = calculate_ik(x_box, y_box, zlist[i], tar_ori_box)
            for j in range(2, 7):
                move_i_robot_arm(j, box_angles[j - 1])
            time.sleep(0.0025)
            if i == 9:
                time.sleep(3)

        time.sleep(7)

        #################
        ## 그리퍼 작동 프로그램
        #################

        for i in range(10):
            zlist = []
            for j in range(1, 11):
                zlist.append(z_box - (z_box - z_hypo_box2) * j / 10)
            box_angles = calculate_ik(x_box, y_box, zlist[i], tar_ori_box)
            for j in range(2, 7):
                move_i_robot_arm(j, box_angles[j - 1])
            time.sleep(0.0025)
            if i == 9:
                time.sleep(3)
        for i in range(1, 7):
            # 속도 제한
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, i, 112, 20)


##########################################################
#########################################################


# =========================
# 단독 실행 테스트용
# =========================

while True:
    coords = get_object_coordinates_for_robot(
            model_path="runs/detect/bowl_ramen_det_50imgs3/weights/best.pt",
            camera_index=0,
            allowed_class_names=["bowl1", "h-ramen"],   # 모든 클래스 허용
            max_frames=60,
            visualize=True,
            merge_threshold_mm=20.0,
    )
    print(len(coords))
    if len(coords) == 0:
        break
    print('coords')
    print(coords)
    print('MODcoords')
    cup_num = len(coords)
    coordsy = []
    for i in range(cup_num):
        coordsy.append(coords[i][1])
    Maxy_index = coordsy.index(max(coordsy))
    x_box = coords[Maxy_index][0] + 150
    y_box = coords[Maxy_index][1] - 250
    Whatbox = coords[Maxy_index][2]

    print(x_box, y_box)
    pick_the_cup(x_box, y_box, 340)
