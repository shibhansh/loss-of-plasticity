import os
import xml.etree.ElementTree as ET

import gym
from gym.utils import EzPickle
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.ant_v3 import AntEnv as AntEnv3
from gym.envs.mujoco.mujoco_env import MujocoEnv


class SlipperyAntEnv(AntEnv, MujocoEnv, EzPickle):
    """
    SlipperyAnt-v2
    """
    def __init__(self, friction=1.0, xml_file='ant.xml'):
        self.xml_file = xml_file
        self.friction = friction
        self.gen_xml_file()
        MujocoEnv.__init__(self, self.xml_file, 5)
        EzPickle.__init__(self)
    
    def gen_xml_file(self):
        old_file = os.path.join(os.path.dirname(gym.envs.mujoco.ant.__file__), "assets", 'ant.xml')
        # Parse old xml file
        tree = ET.parse(old_file)
        root = tree.getroot()
        # Update friction value
        root[3][1].attrib['friction'] = str(self.friction) + ' 0.5 0.5'
        tree.write(self.xml_file)


class SlipperyAntEnv3(AntEnv3, MujocoEnv, EzPickle):
    """
    SlipperyAnt-v3
    """
    def __init__(
        self,
        friction=1.5,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        self.xml_file = xml_file
        self.friction = friction
        self.gen_xml_file()
        
        EzPickle.__init__(**locals())
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        MujocoEnv.__init__(self, self.xml_file, 5)

    def gen_xml_file(self):
        old_file = os.path.join(os.path.dirname(gym.envs.mujoco.ant_v3.__file__), "assets", 'ant.xml')
        # Parse old xml file
        tree = ET.parse(old_file)
        root = tree.getroot()
        # Update friction value
        root[3][1].attrib['friction'] = str(self.friction) + ' 0.5 0.5'
        tree.write(self.xml_file)