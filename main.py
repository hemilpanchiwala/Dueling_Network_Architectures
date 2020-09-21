import utils
import numpy as np

import DuelingDQNAgent as ddqnAgent


if __name__ == '__main__':
    env = utils.make_env('PongNoFrameskip-v4')

    n_games = 500
    scores = []
    epsilon_history = []
    step_count = 0
    best_score = -np.inf

    load_checkpoint = False

    print(env.observation_space.shape)
    agent = ddqnAgent.DuelingDQNAgent(learning_rate=0.0001, n_actions=env.action_space.n,
                                      input_dims=env.observation_space.shape, gamma=0.99,
                                      epsilon=1.0, batch_size=32, memory_size=1000,
                                      replace_network_count=10000,
                                      checkpoint_dir='/home/blackreaper/Documents/temp/duelingdqn/')

    for i in range(n_games):

        obs = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_experience(obs, action, reward, new_obs, int(done))
                agent.learn()
            obs = new_obs
            step_count += 1

        scores.append(score)
        epsilon_history.append(agent.epsilon)
        avg_score = np.mean(scores)

        if score > avg_score:
            if not load_checkpoint:
                agent.save_model()

        if score > best_score:
            best_score = score

        print('episode: ', i, ' score: ', score, ' avg. score: ', avg_score,
              ' best_score: ', best_score, ' epsilon: ', agent.epsilon, ' steps ', step_count)
