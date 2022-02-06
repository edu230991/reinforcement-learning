"""A file to quickly test environments
"""


import gym
import gym_2048

# import gym_wordle
# from gym_wordle.utils import to_array, to_english


def test_2048():
  env = gym.make('2048-v0')
  env.seed(42)

  env.reset()
  env.render()

  done = False
  moves = 0
  while not done:
    action = env.np_random.choice(range(4), 1).item()
    next_state, reward, done, info = env.step(action)
    moves += 1

    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

  print('\nTotal Moves: {}'.format(moves))


def test_wordle():
    # package does not work right now
    env = gym.make('Wordle-v0')

    env.reset()

    done = False

    while not done:
        env.render()
        valid = False

        while not valid:
            guess = input('Guess: ').lower()
            action = to_array(guess)

            if env.action_space.contains(action):
                valid = True

        state, reward, done, info = env.step(action)

    env.render()

    print(f"The word was {to_english(env.solution).upper()}")
