import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, DDPG
import niryo_gym                                
 
def lay_cube_on_floor(env, cube_half=0.015, plane_geom_name="floor0", eps=0.0015):
    m, d = env.unwrapped.model, env.unwrapped.data
    env.unwrapped._mujoco.mj_forward(m, d)

    mj = env.unwrapped._mujoco
    gid_plane = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, plane_geom_name)
    plane_z = float(d.geom_xpos[gid_plane][2]) 

    qpos = env.unwrapped._utils.get_joint_qpos(m, d, "object0:joint").copy()
    qpos[3:] = [1.0, 0.0, 0.0, 0.0]
    qpos[2]  = plane_z + cube_half + eps
    env.unwrapped._utils.set_joint_qpos(m, d, "object0:joint", qpos)

    qvel = env.unwrapped._utils.get_joint_qvel(m, d, "object0:joint").copy()
    qvel[:] = 0.0
    env.unwrapped._utils.set_joint_qvel(m, d, "object0:joint", qvel)

    for _ in range(40):
        mj.mj_step(m, d)

class SettleOnReset(gym.Wrapper):
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        lay_cube_on_floor(self.env, cube_half=0.015) 

        try:
            obs = self.env.unwrapped._get_obs()
        except AttributeError:
            m, d = self.env.unwrapped.model, self.env.unwrapped.data
            self.env.unwrapped._mujoco.mj_forward(m, d)

        return obs, info

if __name__=="__main__":
    # Initialize env and model
    env_name = "NiryoLift-v1"
    env = gym.make(env_name)
    env = SettleOnReset(env) 
    env.reset()            
    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future", # strategies (cf paper): future, final, episode
        ),
        verbose=1,
    )
 
    # Train the model
    model.learn(50000)

    # Visualize trained policy
    vec_env = gym.make(env_name, render_mode="human")
    vec_env = SettleOnReset(vec_env)
    obs, _ = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, trunc, term, info = vec_env.step(action)
        if trunc or term:
            obs, _ = vec_env.reset()
