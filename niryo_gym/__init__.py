from pathlib import Path

ROOT_PATH = Path()

# add other, e.g.
# HPARAM_PATH = ROOT_PATH / 'hparams'

from gymnasium.envs.registration import register

def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    default_kwargs = {
        "reward_type": "sparse",    # one of ['sparse', 'dense']
        "observation_type": "goal"  # one of ['goal', 'camera']
    }

    register(
        id=f"NiryoPickAndPlace-v1",
        entry_point="niryo_gym.niryo_env:MujocoNiryoPickAndPlaceEnv",
        kwargs=default_kwargs,
        max_episode_steps=50,
    )

    register(
        id=f"NiryoLift-v1",
        entry_point="niryo_gym.niryo_env:MujocoNiryoLiftEnv",
        kwargs=default_kwargs,
        max_episode_steps=50,
    )
    
    register(
        id=f"NiryoReach-v1",
        entry_point="niryo_gym.niryo_env:MujocoNiryoReachEnv",
        kwargs=default_kwargs,
        max_episode_steps=50,
    )

register_robotics_envs()