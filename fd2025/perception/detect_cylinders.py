from typing import List, Optional, Union

import networkx as nx
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import ClusterPointIndices
from rpbench.articulated.world.jskfridge import FridgeModel
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton
from scipy.optimize import minimize
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from skrobot.coordinates import Coordinates

from fd2025.perception.magic import POINTCLOUD_OFFSET


def fit_cylinder(points: np.ndarray, z_min, z_max) -> CylinderSkelton:
    def cylinder_sdf_centered(points, radius, height, center):
        x, y, z = (
            points[:, 0] - center[0],
            points[:, 1] - center[1],
            points[:, 2] - center[2],
        )
        d_axis = np.sqrt(x**2 + y**2) - radius
        d_caps = np.abs(z) - height / 2
        return np.maximum(d_axis, d_caps)

    def loss(param: np.ndarray) -> float:
        r_min = 0.02
        r_max = 0.04
        x, y, r = param
        center = np.array([x, y, 0.5 * (z_min + z_max)])
        values = cylinder_sdf_centered(points, r, z_max - z_min, center)
        loss_standard = np.sum(values**2) / len(values)

        if r > r_max:
            loss_standard += loss_standard * 300 * (r - r_max) / 0.01
        if r < r_min:
            loss_standard += loss_standard * 30 * (r_min - r) / 0.01

        # center of mass must be front side of the cylinder
        x_com = np.mean(points[:, 0])
        if x_com - x > 0.0:
            loss_standard *= 3

        return loss_standard

    center = np.mean(points, axis=0)
    param_guess = np.array([center[0], center[1], 0.03])
    ret = minimize(loss, param_guess, method="Nelder-Mead")
    if ret.fun > 0.01:
        ret.success = False
    return ret


def instantiate_fridge_model(param: np.ndarray) -> FridgeModel:
    x, y, yaw, angle = param
    model = FridgeModel(angle)
    model.translate([x, y, 0])
    model.rotate(yaw, "z")
    return model


def model_content_in_fridge(
    points_whole: Union[PointCloud2, np.ndarray],  # PointCloud2 for backward compatibility
    cluster_indicse: ClusterPointIndices,
    fridge_param: np.ndarray,
    known_cylinders: Optional[List[CylinderSkelton]] = None,
) -> List[CylinderSkelton]:

    fridge_model = instantiate_fridge_model(fridge_param)
    attention_region = fridge_model.regions[1].box.detach_clone()
    attention_region.translate(-POINTCLOUD_OFFSET)
    z_lower = attention_region.worldpos()[2] - 0.5 * attention_region._extents[2]

    # Sometimes the points near the wall on the region is detected as a cluster.
    # To avoid this, we shrink the region a little bit. Especially the back side.
    co_shrinked = attention_region.copy_worldcoords()
    extents = attention_region._extents.copy()
    back_side_reduction = 0.08
    co_shrinked.translate([-0.5 * back_side_reduction, 0, 0])
    extents[0] -= back_side_reduction
    extents[1] -= 0.05
    extents[2] -= 0.04
    attention_shrinked = BoxSkeleton(extents=extents)
    attention_shrinked.newcoords(co_shrinked)

    if isinstance(points_whole, PointCloud2):
        gen = pc2.read_points(points_whole, skip_nans=False, field_names=("x", "y", "z"))
        points_whole = np.array(list(gen))
    points_attention = points_whole[attention_shrinked.sdf(points_whole) < 0.0]

    # remove points of known cylinders
    if known_cylinders is not None:
        margin = 0.03
        for cylinder in known_cylinders:
            P = points_attention
            c = cylinder.worldpos()
            is_outside = np.linalg.norm(P[:, :2] - c[:2], axis=1) > cylinder.radius + margin
            points_attention = points_attention[is_outside]
    else:
        known_cylinders = []

    dbscan = DBSCAN(eps=0.01, min_samples=3)
    clusters = dbscan.fit_predict(points_attention)
    n_label = np.max(clusters) + 1

    cylinders: List[CylinderSkelton] = []
    for i in range(n_label):
        sub_points = points_attention[clusters == i]
        if len(sub_points) < 100:
            continue

        z_top_point = np.max(sub_points)
        h = z_top_point - z_lower

        ret = fit_cylinder(sub_points, z_lower, z_top_point)
        r_max = 0.05
        if ret.success:
            x, y, r = ret.x
            if r > r_max:
                r = r_max
            r_margin = 0.003
            p_center = np.array([x, y, z_lower + 0.5 * h])
            cylinder = CylinderSkelton(r + r_margin, h, p_center)
            cylinders.append(cylinder)

    do_refine = True
    if not do_refine:
        return cylinders + known_cylinders
    else:
        positions = np.array([cylinder.worldpos()[:2] for cylinder in cylinders])
        radii = np.array([cylinder.radius for cylinder in cylinders])
        raw_dist_matrix = np.sqrt(((positions[:, np.newaxis] - positions) ** 2).sum(axis=2))
        radii_sum_matrix = radii[:, np.newaxis] + radii

        adj_matrix = raw_dist_matrix < radii_sum_matrix
        np.fill_diagonal(adj_matrix, False)
        adj_matrix = adj_matrix.astype(int)
        G = nx.Graph(adj_matrix)
        meta_clusters = np.zeros(len(cylinders), dtype=int)
        for i, cluster in enumerate(nx.connected_components(G)):
            meta_clusters[list(cluster)] = i

        n_meta_cluster = np.max(meta_clusters) + 1
        cylinders_refined = []
        for i_meta_cluster in range(n_meta_cluster):
            indices_cylinder = np.where(meta_clusters == i_meta_cluster)[0]
            assert len(indices_cylinder) > 0
            assert len(indices_cylinder) < 3
            if len(indices_cylinder) == 1:
                cylinders_refined.append(cylinders[indices_cylinder[0]])
            else:
                # find minimal cylinder containing two cylinders
                cy1 = cylinders[indices_cylinder[0]]
                cy2 = cylinders[indices_cylinder[1]]
                c_xy1 = cy1.worldpos()[:2]
                c_xy2 = cy2.worldpos()[:2]
                dist = np.linalg.norm(c_xy1 - c_xy2)
                r_new = 0.5 * (dist + cy1.radius + cy2.radius)
                c_xy_new = ((c_xy2 - c_xy1) / dist) * (r_new - cy1.radius) + c_xy1
                c_z_new = max(cy1.worldpos()[2], cy2.worldpos()[2])
                height_new = max(cy1.height, cy2.height)
                cylinder_new = CylinderSkelton(r_new, height_new)
                co_new = Coordinates(np.hstack([c_xy_new, c_z_new]), rot=cy1.worldrot())
                cylinder_new.newcoords(co_new)
                cylinders_refined.append(cylinder_new)
        return cylinders_refined + known_cylinders
