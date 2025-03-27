import pickle
import time
from pathlib import Path

import rospy
from sensor_msgs.msg import Image, PointCloud2

if __name__ == "__main__":
    node = rospy.init_node("debug_publisher")
    image_topic = "/kinect_head/rgb/image_rect_color"
    point_cloud_topic = "/local/tf_transform/output"

    msg_log_path = Path(__file__).parent
    replay_datetime = "20250326_082328"
    image_path = msg_log_path / f"image_rect_color-{replay_datetime}.pkl"
    point_cloud_path = msg_log_path / f"point_cloud-{replay_datetime}.pkl"

    with open(image_path, "rb") as f:
        image_msg = pickle.load(f)

    with open(point_cloud_path, "rb") as f:
        point_cloud_msg = pickle.load(f)

    image_pub = rospy.Publisher(image_topic, Image, queue_size=1)
    point_cloud_pub = rospy.Publisher(point_cloud_topic, PointCloud2, queue_size=1)

    while not rospy.is_shutdown():
        image_pub.publish(image_msg)
        point_cloud_pub.publish(point_cloud_msg)
        time.sleep(0.2)
