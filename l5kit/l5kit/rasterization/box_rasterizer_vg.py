from typing import List, Optional, Tuple, Union
import colorsys

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE

from ..data.filter import filter_vehicle_agents_by_labels, filter_nonvehicle_agents_by_labels, filter_agents_by_track_id
from ..geometry import rotation33_as_yaw, transform_points
from .rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import CV2_SUB_VALUES, cv2_subpixel
from .box_rasterizer import get_ego_as_agent # use custom draw_boxes

"""
Key changes wrt box_rasterizer:
  * only show some timestamps (frames_to_plot) vs all of them
  * handling vehicle vs. non-vehicle agents differently by using different color
  * fixing issue with aliased bounding boxes in cv2.fillPoly call in draw_boxes
  * faded bounding boxes overlay and color scheme based on nuscenes-devkit
"""

def draw_boxes(
        raster_size: Tuple[int, int],
        raster_from_world: np.ndarray,
        agents: np.ndarray,
        color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    See draw_boxes in box_rasterizer.py for details.
    This is identical but without the CV2.LINE_AA argument for cv2.fillPoly that causes
    the resulting image to not simply be a binary image / mask.  See link for details:
    https://answers.opencv.org/question/45864/antialiased-polygon-fill-doesnt-respect-area-borders/
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    corners_base_coords = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5)[None, :, :]

    # compute the corner in world-space (start in origin, rotate and then translate)
    corners_m = corners_base_coords * agents["extent"][:, None, :2]  # corners in zero
    s = np.sin(agents["yaw"])
    c = np.cos(agents["yaw"])
    # note this is clockwise because it's right-multiplied and not left-multiplied later,
    # and therefore we're still rotating counterclockwise.
    rotation_m = np.moveaxis(np.array(((c, s), (-s, c))), 2, 0)
    box_world_coords = corners_m @ rotation_m + agents["centroid"][:, None, :2]

    box_raster_coords = transform_points(box_world_coords.reshape((-1, 2)), raster_from_world)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_raster_coords = cv2_subpixel(box_raster_coords.reshape((-1, 4, 2)))
    cv2.fillPoly(im, box_raster_coords, color=color, shift=CV2_SUB_VALUES['shift'])
    return im

class BoxRasterizerVG(Rasterizer):
    def __init__(
            self, render_context: RenderContext, filter_agents_threshold: float, history_num_frames: int, frames_to_plot: np.ndarray
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames in the history.  Note not all of them are rendered.
            frames_to_plot (np.ndarray): Which of the history frames to actually plot to reduce rendering overhead.
        """
        super(BoxRasterizerVG, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        # Check that the indices are valid wrt history_num_frames.  It is 1-indexed s.t. index N = N timesteps prior.
        # Note: the current frame (index = 0) is always plotted and should not be included in frames_to_plot.
        assert np.min(frames_to_plot) > 0 and np.max(frames_to_plot) <= self.history_num_frames        
        self.frames_to_plot = np.sort(frames_to_plot)

        # Using color and fading scheme based on nuscenes, reference here on L112 and L141:
        # https://github.com/nutonomy/nuscenes-devkit/blob/5325d1b400950f777cd701bdd5e30a9d57d2eaa8/python-sdk/nuscenes/prediction/input_representation/agents.py        
        self.agent_vehicle_rgb    = (255, 255, 0)  # yellow
        self.ego_vehicle_rgb      = (255, 0, 0)    # red
        self.agent_nonvehicle_rgb = (255, 153, 51) # orange

        self.min_hsv_value = 0.4                   # minimum value (HSV) for the faded bounding boxes

        self.agent_vehicle_hsv    = colorsys.rgb_to_hsv(*self.agent_vehicle_rgb) 
        self.ego_vehicle_hsv      = colorsys.rgb_to_hsv(*self.ego_vehicle_rgb) 
        self.agent_nonvehicle_hsv = colorsys.rgb_to_hsv(*self.agent_nonvehicle_rgb) 

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # all frames are drawn relative to this one"
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # This ensures we always end up with fixed size arrays.  We plot the current frame + frames_to_plot from history.        
        out_shape = (self.raster_size[1], self.raster_size[0], len(self.frames_to_plot) + 1)
        # Agent images are similar to box_rasterizer.  However, we use color = 255 for vehicle-like agents
        # and color=128 for other relevant but non-vehicle agents (e.g. pedestrians/animals).
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        # Ego images save as in box_rasterizer.
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        channel_index = 0

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            # Downsample to only handle the frames we care about for rendering.
            if i == 0 or i in self.frames_to_plot:
                pass
            else:
                continue
            
            vehicle_agents    = filter_vehicle_agents_by_labels(agents, self.filter_agents_threshold)
            nonvehicle_agents = filter_nonvehicle_agents_by_labels(agents, self.filter_agents_threshold)

            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:                   
                vehicle_agent_image = draw_boxes(self.raster_size, raster_from_world, vehicle_agents, 255)   
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    raise Exception("Expected the dataset to have full history for the rendered agent!")
                    # agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    # ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    vehicle_agents = vehicle_agents[vehicle_agents != agent_ego[0]]
                    vehicle_agent_image = draw_boxes(self.raster_size, raster_from_world, np.append(vehicle_agents, av_agent), 255)
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)
            
            # Add the non-vehicle agents to a combined agent image.
            nonvehicle_agent_image = draw_boxes(self.raster_size, raster_from_world, nonvehicle_agents, 128)             
            unoccupied_mask = (vehicle_agent_image == 0)
            vehicle_agent_image[unoccupied_mask] = nonvehicle_agent_image[unoccupied_mask]
            agents_image = vehicle_agent_image

            agents_images[..., channel_index] = agents_image
            ego_images[..., channel_index] = ego_image
            channel_index += 1
        
        # We combine everything into a single RGB image for simplicity.
        out_im = self._combine_images(agents_images, ego_images)        

        return out_im.astype(np.float32) / 255

    def _combine_images(self, agent_images: np.ndarray, ego_images: np.ndarray) -> np.ndarray:
        out_img = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # I'm making an assumption that the value is the same for all base colors and fixed at 1.
        # This holds for the red, yellow, orange RGB values used in nuscenes.
        val_increment = (1.0 - self.min_hsv_value) / self.history_num_frames

        for ind in reversed( range( len(self.frames_to_plot)+1 ) ): 
            agent_img = agent_images[:, :, ind]
            ego_img   = ego_images[:, :, ind
            ]
            if ind == 0:
                # Current time stamp, use unfaded color.
                vehicle_rgb    = self.agent_vehicle_rgb
                nonvehicle_rgb = self.agent_nonvehicle_rgb
                ego_rgb        = self.ego_vehicle_rgb
            else:
                # History time stamp, use a faded color.
                # Note: we use ind - 1 since index 0 is not in self.frames_to_plot
                timestep = self.frames_to_plot[ind - 1]
                val = 1.0 - val_increment * timestep
                val = int(255 * val)
                
                vehicle_rgb    = colorsys.hsv_to_rgb(self.agent_vehicle_hsv[0], \
                                                     self.agent_vehicle_hsv[1], \
                                                     val)
                nonvehicle_rgb = colorsys.hsv_to_rgb(self.agent_nonvehicle_hsv[0], \
                                                     self.agent_nonvehicle_hsv[1], \
                                                     val)
                ego_rgb        = colorsys.hsv_to_rgb(self.ego_vehicle_hsv[0], \
                                                     self.ego_vehicle_hsv[1], \
                                                     val)

            # Combine by using masks and overlaying on out_img in priority order.
            # Essentially, we plot the most recent information towards the end.
            # And the ego/focused vehicle has precedence over other agents.
            vehicle_img_mask    = (agent_img == 255)
            nonvehicle_img_mask = (agent_img == 128)
            ego_mask            = (ego_img   == 255)

            out_img[vehicle_img_mask]    = vehicle_rgb
            out_img[nonvehicle_img_mask] = nonvehicle_rgb
            out_img[ego_mask]            = ego_rgb

        return out_img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        # We let rasterize handle all the work of making a single combined RGB image.        
        return (in_im * 255).astype(np.uint8) 

    def num_channels(self) -> int:
        return 3
