import numpy as np

# NOTE: do not consider this offset in grasping target detection. Grasping usually require high precision knowledge of the relative target pose to the gripper.
# Therefore, instead of using a fixed offset value, we will dynamically estimate the offset using the yellow tape in the execution phase.
# To the dynamically estimation be valid, any estimation using points should not be calibrated in the recording phase.
POINTCLOUD_OFFSET = -np.array(
    [
        -0.02164367,
        -0.01797294,
        0.01358793,
    ]  # obtained by running ./fridge_demo/calibrator.py
)  # offset of kinematic model from point cloud. You may add this offset to the point cloud to match the kinematic model.

TOUCH_LOCALIZTION_OBJECT_HALF_WIDTH = 0.05  # only vaild for mugcup
ISHIDA_DESKTOP_MAX_PROCESS = (
    8  # in my setting, some core are ignored in out-of linux process scheduling...
)
MUG_TOUCH_Z_OFFSET = 0.06  # only vaild for mugcup touching operation
