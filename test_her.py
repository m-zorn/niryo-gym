import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, DDPG

if __name__=="__main__":
    # Initialize env and model
    env_name = "NiryoReach-v1"
    env = gym.make(env_name)
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
    obs, _ = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, trunc, term, info = vec_env.step(action)
        if trunc or term:
            obs, _ = vec_env.reset()