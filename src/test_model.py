import os
import gym
import gym_2048

from stable_baselines3 import PPO


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_and_save(
    env_name = 'LunarLander-v2',
    policy_name = 'MlpPolicy',
    total_timesteps = 200000,
):
    env = gym.make(env_name)
    env.reset()

    model = PPO(policy_name, env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{MODELS_DIR}/{env_name}/{total_timesteps}")


def play(
    env_name = 'LunarLander-v2',
    total_timesteps = 200000,
    n_episodes = 2,
):

    env = gym.make(env_name)  # continuous: LunarLanderContinuous-v2
    env.reset()

    model_path = f"{MODELS_DIR}/{env_name}/{total_timesteps}.zip"
    model = PPO.load(model_path, env=env)

    for ep in range(n_episodes):
        print("%"*50)
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            print(rewards)


if __name__ == "__main__":
    #train_and_save('2048-v0')
    play()