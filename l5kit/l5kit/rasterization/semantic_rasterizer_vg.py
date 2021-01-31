from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import colorsys

import cv2
import numpy as np

from ..data.filter import filter_tl_faces_by_status
from ..data.map_api import InterpolationMethod, MapAPI, TLFacesColors
from ..geometry import rotation33_as_yaw, transform_point, transform_points
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import CV2_SUB_VALUES, CV2_SHIFT_VALUE, INTERPOLATION_POINTS, RasterEls, indices_in_bounds, cv2_subpixel

COLORS = {
    TLFacesColors.GREEN.name: (0, 128, 0),
    TLFacesColors.RED.name: (128, 0, 0),
    TLFacesColors.YELLOW.name: (128, 128, 0),
    RasterEls.LANE_NOTL.name: (255, 217, 82),
    RasterEls.ROAD.name: (255, 255, 255),
    RasterEls.CROSSWALK.name: (119, 136, 153),
}

def angle_to_color(angle: float) -> Tuple[int, int, int]:
    angle = angle + np.pi
    angle = np.degrees(angle)

    color = colorsys.hsv_to_rgb( angle/360, 1., 1.)    
    color = [int(255 * c) for c in color]
    
    return tuple(color)


class SemanticRasterizerVG(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
            self, render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray,
    ):
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        self.mapAPI = MapAPI(semantic_map_path, world_to_ecef)

        # Setting a radius for plotting traffic lights.  Using 1.2 m as in carla_birdeye_view:
        # https://github.com/deepsense-ai/carla-birdeye-view/blob/f666b0f5c11e9a3eb29ea2f9465d6b5526ab1ae0/carla_birdeye_view/mask.py#L329        
        self.tl_radius = np.round(1.2 / self.pixel_size[0], 0).astype(np.int)        

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]
        
        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
            self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        img = np.zeros(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # get all lanes as interpolation so that we can transform them all together
        lane_indices = indices_in_bounds(center_in_world, self.mapAPI.bounds_info["lanes"]["bounds"], raster_radius)

        # Lanes mask with keys given by the traffic light state and values indicating which lanes are affected
        # by that traffic light state (NOTL, RED, GREEN, YELLOw).
        lanes_mask: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool))

        # This contains the left and right boundaries of the lanes..
        lanes_area  = np.zeros((len(lane_indices) * 2, INTERPOLATION_POINTS, 2))
        centerlines = np.zeros((len(lane_indices), INTERPOLATION_POINTS, 2))

        # Traffic light locations and states for plotting.
        tl_states = []

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]

            # interpolate over polyline to always have the same number of points
            lane_coords = self.mapAPI.get_lane_as_interpolation(
                lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
            
            # Id the lane boundary and centerline for later plotting.
            lanes_area[idx * 2] = lane_coords["xyz_left"][:, :2]
            lanes_area[idx * 2 + 1] = lane_coords["xyz_right"][::-1, :2]
            centerlines[idx] = lane_coords["xyz_midlane"][:, :2]

            # Find active and nearby traffic lights for plotting.
            # lanes_mask used to segregate lanes by traffic light state (unused).            
            lane_type = RasterEls.LANE_NOTL.name
            lane_tl_ids = set(self.mapAPI.get_lane_traffic_control_ids(lane_idx))

            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                lane_type  = self.mapAPI.get_color_for_face(tl_id)
                tl_element = self.mapAPI[tl_id].element.traffic_control_element
                tl_xyz     = self.mapAPI.unpack_deltas_cm(tl_element.points_x_deltas_cm,
                                                          tl_element.points_y_deltas_cm,
                                                          tl_element.points_z_deltas_cm,
                                                          tl_element.geo_frame)
                tl_xy_center = np.expand_dims(np.mean(tl_xyz, axis=0)[:2], 0)
                tl_xy = np.round(transform_points(tl_xy_center[:,:2], raster_from_world), 0).astype(np.int)
                tl_states.append( [*tl_xy[0], lane_type] )                

            lanes_mask[lane_type][idx * 2: idx * 2 + 2] = True

        # plot lane areas
        if len(lanes_area):
            lanes_area = cv2_subpixel(transform_points(lanes_area.reshape((-1, 2)), raster_from_world))

            for lane_area in lanes_area.reshape((-1, INTERPOLATION_POINTS * 2, 2)):
                # need to for-loop otherwise some of them are empty
                cv2.fillPoly(img, [lane_area], COLORS[RasterEls.ROAD.name], **CV2_SUB_VALUES)

            # Do dilation on the lane masks since neighboring lanes may have a small gap between them.
            img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)

            # Not using this for now, but could be nice if you want to draw lanes with TL color.
            # lanes_area = lanes_area.reshape((-1, INTERPOLATION_POINTS, 2))
            # for name, mask in lanes_mask.items():  # draw each type of lane with its own color
            #     cv2.polylines(img, lanes_area[mask], False, COLORS[name], **CV2_SUB_VALUES)

        # plot crosswalks        
        for idx in indices_in_bounds(center_in_world, self.mapAPI.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.mapAPI.get_crosswalk_coords(self.mapAPI.bounds_info["crosswalks"]["ids"][idx])
            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
            cv2.fillPoly(img, [xy_cross], COLORS[RasterEls.CROSSWALK.name], **CV2_SUB_VALUES)

        # plot traffic lights
        for tl_state in tl_states:            
            cv2.circle(img, tuple(tl_state[:2]), self.tl_radius, COLORS[tl_state[2]], -1) # TL visualized.

        # plot lane centerlines
        for centerline in centerlines:            
            pix_cl = transform_points(centerline[:, :2], raster_from_world)

            for start_px, end_px in zip(pix_cl[:-1], pix_cl[1:]):
                d_px = end_px - start_px
                angle_px = np.arctan2(-d_px[1], d_px[0]) # minus sign since image coord y-axis is downward facing
                color = angle_to_color(angle_px)

                start_px = tuple( np.round(start_px, 0).astype(np.int) )
                end_px   = tuple( np.round(end_px, 0).astype(np.int) )
                cv2.line(img, start_px, end_px, color, thickness = 5)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)

    def num_channels(self) -> int:
        return 3
