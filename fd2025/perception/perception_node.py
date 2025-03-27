import datetime
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from detic_ros.msg import SegmentationInfo
from detic_ros.srv import DeticSeg, DeticSegResponse
from rpbench.articulated.pr2.jskfridge import AV_INIT
from rpbench.articulated.world.jskfridge import FridgeModel
from rpbench.articulated.world.utils import CylinderSkelton
from sensor_msgs.msg import Image, PointCloud2
from skrobot.models import PR2
from skrobot.viewers import PyrenderViewer

from fd2025.perception.detect_cylinders import model_content_in_fridge
from fd2025.perception.detect_fridge import FridgeModelReconciler
from fd2025.ros_numpy.point_cloud2 import pointcloud2_to_xyz_array


@dataclass
class FridgeEnvDetection:
    fridge_param: np.ndarray
    cylinders: List[CylinderSkelton]

    def visualize(self, v: Optional[PyrenderViewer] = None) -> PyrenderViewer:
        if v is None:
            v = PyrenderViewer()
        pr2 = PR2()
        pr2.angle_vector(AV_INIT)
        v.add(pr2)

        # instantiate fridge
        x, y, yaw, angle = self.fridge_param
        fridge = FridgeModel(angle)
        fridge.translate([x, y, 0])
        fridge.rotate(yaw, "z")
        fridge.add(v)

        # instantiate cylinders
        for c in self.cylinders:
            v.add(c.to_visualizable())
        return v


class PerceptionNodeBase:
    _is_active: bool
    _save_msg: bool
    _datetime: Optional[datetime.datetime]
    _mask: Optional[np.ndarray]
    _points: Optional[np.ndarray]
    _cache: Optional[FridgeEnvDetection]
    _segment_image_fn: DeticSeg
    _lock: Lock

    def image_callback(self, msg: Image):
        if not self._is_active or self._mask is not None:
            return

        if self._save_msg:
            assert self._datetime is not None
            name = f"image_rect_color-{self._datetime.strftime('%Y%m%d_%H%M%S')}.pkl"
            rospy.loginfo(f"Saving image to {name}")
            file_path = Path(__file__).parent / "msg_log" / name
            with open(file_path, "wb") as f:
                pickle.dump(msg, f)

        try:
            response: DeticSegResponse = self._segment_image_fn(msg)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        seg_info: SegmentationInfo = response.seg_info
        bridge = CvBridge()
        self._mask = bridge.imgmsg_to_cv2(seg_info.segmentation, desired_encoding="passthrough")
        with self._lock:
            self._check_and_process_data()

    def point_cloud_callback(self, msg: PointCloud2):
        if not self._is_active or self._points is not None:
            return

        if self._save_msg:
            assert self._datetime is not None
            name = f"point_cloud-{self._datetime.strftime('%Y%m%d_%H%M%S')}.pkl"
            rospy.loginfo(f"Saving point cloud to {name}")
            file_path = Path(__file__).parent / "msg_log" / name
            with open(file_path, "wb") as f:
                pickle.dump(msg, f)

        points = pointcloud2_to_xyz_array(msg, remove_nans=False).reshape(-1, 3)
        self._points = points

        with self._lock:
            self._check_and_process_data()

    def _check_and_process_data(self):
        if self._mask is None or self._points is None:
            return
        rec = FridgeModelReconciler()
        fridge_param = rec.reconcile(self._points[self._mask.flatten() == 1])
        cylinders_cand = model_content_in_fridge(self._points, None, fridge_param)
        # remove cylinder with small height
        cylinders = [c for c in cylinders_cand if c.height > 0.05]
        self._cache = FridgeEnvDetection(fridge_param, cylinders)
        self._is_active = False


class PerceptionNode(PerceptionNodeBase):
    def __init__(self, save_msg: bool = False):
        self._is_active = False
        self._mask = None
        self._points = None
        image_topic = "/kinect_head/rgb/image_rect_color"
        point_cloud_topic = "/local/tf_transform/output"
        rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.Subscriber(point_cloud_topic, PointCloud2, self.point_cloud_callback)

        self._cache = None
        self._datetime = datetime.datetime.now()
        self._save_msg = save_msg

        service_name = "/local/detic_segmentor/segment_image"
        rospy.wait_for_service(service_name)
        self._segment_image_fn = rospy.ServiceProxy(service_name, DeticSeg, persistent=True)
        self._lock = Lock()

    def percept(self) -> FridgeEnvDetection:
        self._cache = None
        self._mask = None
        self._points = None
        self._is_active = True
        while self._cache is None:
            time.sleep(0.02)
        return self._cache


class PerceptionDebugNode(PerceptionNodeBase):
    def __init__(self, replay_datetime: str):
        self._save_msg = False  # No need to save msg in debug mode
        data_dir = Path(__file__).parent / "msg_log"
        image_topic_path = data_dir / f"image_rect_color-{replay_datetime}.pkl"
        self._lock = Lock()
        with image_topic_path.open(mode="rb") as f:
            image_msg = pickle.load(f)
        point_cloud_topic_path = data_dir / f"point_cloud-{replay_datetime}.pkl"
        with point_cloud_topic_path.open(mode="rb") as f:
            point_cloud_msg = pickle.load(f)

        service_name = "/local/detic_segmentor/segment_image"
        rospy.wait_for_service(service_name)
        self._segment_image_fn = rospy.ServiceProxy(service_name, DeticSeg, persistent=True)

        self._is_active = True  # No need to wait for data
        self._mask = None
        self._points = None

        self.image_callback(image_msg)
        self.point_cloud_callback(point_cloud_msg)

    def percept(self) -> FridgeEnvDetection:
        return self._cache


if __name__ == "__main__":
    rospy.init_node("perception_node")
    # node = PerceptionNode(save_msg=False, replay_datetime="20250326_082328")
    # node = PerceptionNode(save_msg=True)
    # node = PerceptionNode(save_msg=False)
    node = PerceptionDebugNode(replay_datetime="20250326_082328")
    detection = node.percept()
    v = detection.visualize()
    v.show()
    time.sleep(1000)
