import gymnasium as gym
from stable_baselines3 import PPO
import niryo_gym

if __name__=="__main__":
    # Initialize env and model
    env_name = "NiryoReach-v1"
    env = gym.make(env_name, render_mode="human", observation_type="camera")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
    )

    # Train the model
    model.learn(50000)

    # Visualize trained policy
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, trunc, term, info = env.step(action)
        if trunc or term:
            obs, _ = env.reset()