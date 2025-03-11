from setuptools import find_packages, setup

setup(
    name='niryo_gym',
    packages=find_packages(),
    version='0.1.0',
    description="Reinforcement learning mujoco gym-environment for the Niryo-NED2 robot arm, based on Farama foundation's Fetch robots.",
    author='Maximilian Zorn',
    license="MIT"
)