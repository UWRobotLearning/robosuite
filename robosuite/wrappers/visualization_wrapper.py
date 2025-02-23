"""
This file implements a wrapper for visualizing important sites in a given environment.

By default, this visualizes all sites possible for the environment. Visualization options
for a given environment can be found by calling `get_visualization_settings()`, and can
be set individually by calling `set_visualization_setting(setting, visible)`.
"""
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.wrappers import Wrapper

DEFAULT_INDICATOR_SITE_CONFIG = {
    "type": "sphere",
    "size": [0.03],
    "rgba": [1, 0, 0, 0.5],
}


class VisualizationWrapper(Wrapper):
    def __init__(self, env, indicator_configs=None):
        """
        Initializes the data collection wrapper. Note that this automatically conducts a (hard) reset initially to make
        sure indicators are properly added to the sim model.

        Args:
            env (MujocoEnv): The environment to visualize

            indicator_configs (None or str or dict or list): Configurations to use for indicator objects.

                If None, no indicator objects will be used

                If a string, this should be `'default'`, which corresponds to single default spherical indicator

                If a dict, should specify a single indicator object config

                If a list, should specify specific indicator object configs to use for multiple indicators (which in
                turn can either be `'default'` or a dict)

                As each indicator object is essentially a site element, each dict should map site attribute keywords to
                values. Note that, at the very minimum, the `'name'` attribute MUST be specified for each indicator. See
                http://www.mujoco.org/book/XMLreference.html#site for specific site attributes that can be specified.
        """
        super().__init__(env)

        # Make sure that the environment is NOT using segmentation sensors, since we cannot use segmentation masks
        # with visualization sites simultaneously
        assert all(
            seg is None for seg in env.camera_segmentations
        ), "Cannot use camera segmentations with visualization wrapper!"

        # Standardize indicator configs
        self.indicator_configs = None
        if indicator_configs is not None:
            self.indicator_configs = []
            if type(indicator_configs) in {str, dict}:
                indicator_configs = [indicator_configs]
            for i, indicator_config in enumerate(indicator_configs):
                if indicator_config == "default":
                    indicator_config = deepcopy(DEFAULT_INDICATOR_SITE_CONFIG)
                    indicator_config["name"] = f"indicator{i}"
                # Make sure name attribute is specified
                assert "name" in indicator_config, "Name must be specified for all indicator object configurations!"
                # Add this configuration to the internal array
                self.indicator_configs.append(indicator_config)

        # Create internal dict to store visualization settings (set to True by default)
        self._vis_settings = {vis: True for vis in self.env._visualizations}

        # Add the post-processor to make sure indicator objects get added to model before it's actually loaded in sim
        self.env.set_xml_processor(processor=self._add_indicators_to_model)

        # Conduct a (hard) reset to make sure visualization changes propagate
        reset_mode = self.env.hard_reset
        self.env.hard_reset = True
        self.reset()
        self.env.hard_reset = reset_mode

    def get_indicator_names(self):
        """
        Gets all indicator object names for this environment.

        Returns:
            list: Indicator names for this environment.
        """
        return (
            [ind_config["name"] for ind_config in self.indicator_configs] if self.indicator_configs is not None else []
        )

    def set_indicator_pos(self, indicator, pos):
        """
        Sets the specified @indicator to the desired position @pos

        Args:
            indicator (str): Name of the indicator to set
            pos (3-array): (x, y, z) Cartesian world coordinates to set the specified indicator to
        """
        # Make sure indicator is valid
        indicator_names = set(self.get_indicator_names())
        assert indicator in indicator_names, "Invalid indicator name specified. Valid options are {}, got {}".format(
            indicator_names, indicator
        )
        # Set the specified indicator
        self.env.sim.model.body_pos[self.env.sim.model.body_name2id(indicator + "_body")] = np.array(pos)

    def get_visualization_settings(self):
        """
        Gets all settings for visualizing this environment

        Returns:
            list: Visualization keywords for this environment.
        """
        return self._vis_settings.keys()

    def set_visualization_setting(self, setting, visible):
        """
        Sets the specified @setting to have visibility = @visible.

        Args:
            setting (str): Visualization keyword to set
            visible (bool): True if setting should be visualized.
        """
        assert (
            setting in self._vis_settings
        ), "Invalid visualization setting specified. Valid options are {}, got {}".format(
            self._vis_settings.keys(), setting
        )
        self._vis_settings[setting] = visible

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate visualization

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        # Update any visualization
        self.env.visualize(vis_settings=self._vis_settings)
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate visualization

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)

        # Update any visualization
        self.env.visualize(vis_settings=self._vis_settings)

        return ret

    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        if self.indicator_configs is not None:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")

            for indicator_config in self.indicator_configs:
                config = deepcopy(indicator_config)
                indicator_body = new_body(name=config["name"] + "_body", pos=config.pop("pos", (0, 0, 0)))
                indicator_body.append(new_site(**config))
                worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")

        return xml

    def create_trajectory_visualization(self, points, rgba=(0, 0, 1, 0.5), size=0.02):
        """
        Creates visualization sites for trajectory points.
        
        Args:
            points (np.ndarray): Nx3 array of trajectory points
            rgba (tuple): Color and transparency (default: semi-transparent red)
            size (float): Radius of trajectory spheres
        """
        # Reference the relevant XML structure
        root = ET.fromstring(self.env.sim.model.get_xml())
        worldbody = root.find("worldbody")
        # remove existing trajectory points
        for site in worldbody.findall(".//site[@name^='traj_point_']"):
            worldbody.remove(site)

        # Create a visualization site for each point
        for i, point in enumerate(points):
            site_name = f"traj_point_{i}"
            traj_site = ET.Element("site",
                name=site_name,
                pos=f"{point[0]} {point[1]} {point[2]}",
                size=f"{size}",
                rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
                type="sphere",
                group="1"
            )
            worldbody.append(traj_site)
        
        # Update the simulation with new XML
        xml_str = ET.tostring(root, encoding='unicode')
        self.env.reset_from_xml_string(xml_str)

    def create_trajectory_capsules(self, points, rgba=(1, 0, 0, 0.5), radius=0.005):
        """
        Creates a smooth trajectory using connected capsules.
        
        Args:
            sim (MjSim): MuJoCo simulation instance
            points (np.ndarray): Nx3 array of trajectory points
            rgba (tuple): Color and transparency
            radius (float): Radius of the trajectory line
        """
        root = ET.fromstring(self.env.sim.model.get_xml())
        worldbody = root.find("worldbody")
        for geom in worldbody.findall(".//geom[@name^='traj_capsule_']"):
            worldbody.remove(geom)
    
        # Create capsules between consecutive points
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            
            # Calculate capsule position and size
            pos = (p1 + p2) / 2
            direction = p2 - p1
            dist = np.linalg.norm(direction)
            
            if dist < 1e-6:  # Skip if points are too close
                continue
                
            # Calculate rotation quaternion
            direction = direction / dist  # normalize
            # Find rotation that aligns z-axis with direction vector
            z_axis = np.array([0, 0, 1])
            
            if np.allclose(direction, z_axis):
                quat = np.array([1, 0, 0, 0])
            elif np.allclose(direction, -z_axis):
                quat = np.array([0, 1, 0, 0])
            else:
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(z_axis, direction))
                quat = np.array([np.cos(angle/2),
                            rotation_axis[0] * np.sin(angle/2),
                            rotation_axis[1] * np.sin(angle/2),
                            rotation_axis[2] * np.sin(angle/2)])
            
            # Create capsule geom
            geom = ET.Element("geom",
                name=f"traj_capsule_{i}",
                type="capsule",
                pos=f"{pos[0]} {pos[1]} {pos[2]}",
                quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                size=f"{radius} {dist/2}",  # radius and half-length
                rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
                contype="0",
                conaffinity="0",
                group="1"
            )
            worldbody.append(geom)
        
        xml_str = ET.tostring(root, encoding='unicode')
        self.env.reset_from_xml_string(xml_str)
        
    def create_trajectory_spline(self, points, rgba=(1, 0, 0, 0.5), radius=0.005, segments_per_point=5):
        """
        Creates a smooth spline trajectory visualization.
        
        Args:
            sim (MjSim): MuJoCo simulation instance
            points (np.ndarray): Nx3 array of trajectory points
            rgba (tuple): Color and transparency
            radius (float): Radius of the trajectory line
            segments_per_point (int): Number of segments between each pair of points
        """
        from scipy.interpolate import CubicSpline
        
        # Create parameter for spline (cumulative distance along trajectory)
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            t[i] = t[i-1] + np.linalg.norm(points[i] - points[i-1])
        
        # Create spline for each dimension
        splines = [CubicSpline(t, points[:, i]) for i in range(3)]
        
        # Generate smoother point set
        t_new = np.linspace(t[0], t[-1], len(points) * segments_per_point)
        smooth_points = np.vstack([spl(t_new) for spl in splines]).T
        
        # Create visualization using the smooth points
        self.create_trajectory_capsules(smooth_points, rgba, radius)