# L5Kit Notes

I am working with version 1.2.0.  This version includes prediction + planning notebooks and support for dynamic traffic lights.

---
## data_format.md
  * see this for details on the numpy structured array/zarr format.
  * Lyft Competition format
    * `scenes`: includes details of ego vehicle, start time, end time, and corresponding interval in frames.
    * `frames`: like a snapshot, contains timestamp, ego pose, and pointers to relevant agents (vehicles, cyclists, pedestrians) and traffic lights affecting visible lanes.
    * `agents`: contains tracking info like bbox info, velocity, id, and classification probabilities.
    * `tl_faces`: while semantic map has the static information about traffic lights (e.g. locations), this contains the dynamic information for things like which light bulb is active at a given traffic light.
  * ChunkedDataset used to access the zarr dataset efficiently.
  * Helper classes inheriting from Pytorch Dataset are `EgoDataset` and `AgentDataset`.
    * `EgoDataset` only includes ego as the vehicle for which prediction is done.
    * `AgentDataset` additionally includes the detected agents, although you need to provide a mask (e.g. minimal detection threshold, subset to evaluate, etc.).
    * For both approaches, you provide a config and rasterizer which results in the dataset producing instances of dict with the following keys:
      * track_id: -1 for ego, else id for tracked agent
      * timestamp
      * {centroid, yaw, extent}: object properties in the world reference system
      * image: BEV raster
      * target_{positions, yaws, availabilities}: future information in agent frame
      * history_{positions, yaws, availabilities}: past information in agent frame
      * X_from_Y, where X, Y are an element of {raster, world, agent}. 
---

## coords_system.md
Ignored satellite coordinate system, since I'll focus on the semantic coordinate system.

### World Coordinate System
  * Ego translation is XYZ about a fixed origin in Palo Alto.  
  * Ego orientation is a 3x3 rotation matrix with angles measured ccw.  
  * Other agents only have XY and yaw information.

### Agent Coordinate System
  * Agent space is set up so that the ego vehicle is located at (0,0) and ego's heading is 0.  So the vehicle points to the right, aligned with the agent x-axis.
  * Similar to the raster image aside from a simple meters_to_pixel transform.

### Image Coordinate System
  * 2D pixel space where (0,0) is the top left corner.
  * Some good examples of how to transform points using the API:
   * `positions_in_world = transform_points(data["target_positions"], data["world_from_agent"]`
   * `positions_in_raster = transform_points(positions_in_world, data["raster_from_world"])`
   * There is also a helpful draw_trajectory function.

### Semantic Coordinate System
  * The semantic map is stored in a protobuf file (.pb).
  * Contains layers for things like lanes, crosswalks, traffic lights.
  * Basically each feature has its own local ENU coordinates for things like crosswalk polygons, lane edges, etc.  This is paired with transforms to get into world/ECEF frames.
  * Rely on the `MapAPI` class to handle these transforms for you.
---

## competition.md
  * For the prediction challenge, the goal is to produce K trajectory (XY) hypotheses with an associated probability.
  * The main metric used is a negative log likelihood with the following assumptions:
    * x, y are independent
    * the states are independent in time as well (i.e. x_k and x_j are independent for j != k)
    * fixed unit variance for x and y.
  * The ground truth is given in the world coordinate frame so you need to convert them first.
  * ADE/FDE are also good metrics to use.  Average over the hypotheses or take the best one (oracle).
---

## l5kit.data
* Includes the map api, which is good to understand how to extract from static layers.
* See proto definitions in road_network_pb2.py, which includes things like:
  * road/junction/lane/segment information + connectivity
  * crosswalk information
  * traffic control elements (e.g. stop/yield signs, stop lines, traffic lights)
  * These can be added to the semantic_rasterizer in the main for-loop in the get_bounds function.  That loops over all map elements so just add the ones you care about there.
---

## l5kit.dataset
* Implements both `AgentDataset` and `EgoDataset`.
* The key parameters are perturbation (e.g. for planning), config (for rasterization/horizons), dataset, and the rasterizer.
* The key function used is a partial function wrapper around generate_agent_sample.
* For the `AgentDataset`, the filter_agents_threshold is important for the masking.  There are also some default thresholds specified for things like distance to ego.  Otherwise inherits from `EgoDataset` and is pretty similar.
---

## l5kit.geometry
* Some useful angle / oriented crop utils.
* Transform utils to go between frames, like agent -> world.
---

## l5kit.kinematic
* Defines kinematic perturbations which can help to augment imitation learning datasets.
* The key aspect of this is solving a nonlinear least squares problem which balances trajectory tracking with kinematic feasibility.
* The dynamics assume that yaw and velocity are control inputs for an approximate version and that steering/acceleration are the control inputs for the exact variant.
* Use the Gauss-Newton algorithm, which iteratively solves a set of least squares problems where A = jacobian and b = model fit residuals.
* The actual perturbation is done by computing random e_y and e_psi errors to the initial state.
* Then the fit basically tries to get the vehicle to go back to the original trajectory from the perturbed initial state.
---

## l5kit.evaluation
* Handles things like writing predictions to csv with and without ground truth.
* Computes metrics given a ground truth and predicted set of csv files.
* Metrics involve function calls with the following signature.
  * 4 Inputs: ground_truth, pred, confidence, avails (since ground_truth may not be valid for all timesteps)
  * Metric Output as single array or scalar.
* Metrics include (for multimodal, fixed variance predictions):
  * negative log-likelihood (nll)
  * root mean squared error from nll
  * average/final displacment error with best hypothesis (oracle) or the mean over all.
---

## l5kit.rasterization
* build_rasterizer: Takes in the provided config and chooses which of the following Rasterizers to use, along with the settings required.  It also does the parsing of the semantic/aerial map.

### Stub Rasterizer
* Just returns a black image, more for testing.

### Box Rasterizer
* Renders the faded box history for a specified agent (else ego by default).
* Handles all the oriented bbox plotting
* Keeps a target agent (like ego) and other agents in a set of historical images (a_t, a_t-1, a_t-H, e_t, e_t-1, e_t-H).  So it results in (H+1) * 2 frames.
* RGB version involves fading colors and different hues for ego vs. non-ego agents.  Good to compare against nuScenes rasterizer.

### SemanticRasterizer
* Handles all map-related layers (lane and crosswalks) and traffic light state at the current timestep of interest.
* Changes the lane polyline color based on which traffic light is active.  Uses the MapAPI and dynamic traffic light indices to see what is affecting the agent coming from that lane.
* One challenge is how these traffic lights are annotated.  If it's a function of it being in ego's view, then it is a potentially noisy input.
* Uses white for background and darker color for roads, sort of flip of nuScenes. 

### SatelliteRasterizer
* Similar to SemanticRasterizer but it uses the aerial satellite imagery as the layer rasterized.

### {Sat/Sem}BoxRasterizer
* Combines multiple rasterizers into one.
* The box rasterizer has precedence, meaning agents are the "foreground" and the other layer is the "background".
---

## l5kit.sampling
* Gets the history/future context for agents.
* Utils for things like agent velocity and relative pose.
* generate_agent_sample generates a single dataset instance as a dict, key for the dataset implementations.  This is the main building block for the prediction datasets.
---

## l5kit.visualization
* Useful utils for things like overlaying the trajectory on the image.  Worth revisiting when needed.
* Also utils to write videos/gifs.
---

## l5kit.planning
* Baseline model is ResNet + a single linear layer.  Predicts trajectory directly with specified loss function.
* It also provides utils for detecting collisions.  It just involves bounding box intersections and using the edges to determine the direction of the collision.
