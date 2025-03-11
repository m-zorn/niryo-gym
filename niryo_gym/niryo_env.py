import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from niryo_gym.farama_robotics.robot_env import MujocoRobotEnv
from niryo_gym.farama_robotics import rotations

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "niryo_robot/mjcf/ned2/ned2.xml")

DEFAULT_CAMERA_CONFIG = {
    # "distance": 1.5,
    # "azimuth": 200.0,
    # "elevation": -14.0,
    # "lookat": np.array([1.3, 0.75, 0.55]),
    "type":2,
    "fixedcamid": 1, 
}

# type       => 0: free, 1: tracking, 2: fixed
# fixedcamid => 0: gripper POV, 1: full-frontal
POV_CAMERA_CONFIG = {
    "type":2,
    "fixedcamid": 0, 
}

FRONT_CAMERA_CONFIG = {
    "type":2, 
    "fixedcamid": 1,
}

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BaseNiryoEnv(MujocoRobotEnv):
    """Superclass for all Niryo environments."""
    
    metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": 50,
        }

    def __init__(
        self,
        gripper_extra_height,
        block_gripper,
        has_object: bool,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        reward_type,
        **kwargs,
    ):
        """Initializes a new Niryo environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super().__init__(**kwargs) #n_actions=4

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        return action

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )
        
        observation = {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }
        
        if self.observation_type == "camera":
            render_mode="rgb_array"
            view = self.mujoco_renderer._get_viewer(render_mode=render_mode)
            
            observation.update({
                "camera_pov": view.render(render_mode=render_mode, camera_id=POV_CAMERA_CONFIG["fixedcamid"]),
                "camera_front": view.render(render_mode=render_mode, camera_id=FRONT_CAMERA_CONFIG["fixedcamid"]),
            })
        
        return observation

    def generate_mujoco_observations(self):

        raise NotImplementedError

    def _get_gripper_xpos(self):

        raise NotImplementedError

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

class MujocoNiryoEnv(BaseNiryoEnv):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = (
                np.zeros(0)
            )
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        # Reset buffers for joint states, actuators, warm-start, control buffers etc.
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] += object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]


class MujocoNiryoReachEnv(MujocoNiryoEnv, EzPickle):

    def __init__(self, reward_type="sparse", observation_type="goal", **kwargs):
        # position data is 7 numbers (3D position followed by unit quaternion),
        # velocity data is 6 numbers (3D linear velocity followed by 3D angular velocity)
        self.observation_type = observation_type
        initial_qpos = {
            # "robot0:base_link": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "object0:joint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoNiryoEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            n_actions=4,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, observation_type=observation_type, **kwargs)

class MujocoNiryoPickAndPlaceEnv(MujocoNiryoEnv, EzPickle):

    def __init__(self, reward_type="sparse", observation_type="goal", **kwargs):
        # position data is 7 numbers (3D position followed by unit quaternion),
        # velocity data is 6 numbers (3D linear velocity followed by 3D angular velocity)
        self.observation_type = observation_type
        initial_qpos = {
            # "robot0:base_link": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "object0:joint": [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoNiryoEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            n_actions=4,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

class MujocoNiryoLiftEnv(MujocoNiryoEnv, EzPickle):

    def __init__(self, reward_type="sparse", observation_type="goal", **kwargs):
        # position data is 7 numbers (3D position followed by unit quaternion),
        # velocity data is 6 numbers (3D linear velocity followed by 3D angular velocity)
        self.observation_type = observation_type
        initial_qpos = {
            # "robot0:base_link": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "object0:joint": [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoNiryoEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=[0.5, 0, 0],
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            n_actions=4,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        if self.has_object:
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            
            goal = object_qpos[:3]
            goal[2] = 0.25 
        else:
            raise ValueError("Lift Environment needs a Object! Make sure `has_object = True`!")
        return goal.copy()