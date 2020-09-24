# Dueling-Network-Architectures
Implementation of Dueling Network Architectures for Deep Reinforcement Learning paper with Pytorch

Link to the paper: https://arxiv.org/pdf/1511.06581.pdf

# Setup
## Prerequirements
- Python 
- Numpy 
- OpenAI gym
- Pytorch
- Matplotlib (for plotting the results)
For installing all the requirements just run the requirements.txt file using the following command:
```
pip3 install -r requirements.txt
```
# Running the pre-trained model
- Just run the main.py file by the command `python main.py` with making the `load_checkpoint` variable to `True` which will load the saved parameters of the model and output the results.

# Training the model
- You can train the model by running the main.py file using the command
 ```
 python main.py
 ```
- You can set the number of games for training by changing the value of variable `n_games` (current is `500`) in `main.py`.
- You can set the hyper-parameters such as learning_rate, discount factor (gamma), epsilon, and others while initializing the `agent` variable in `main.py`.
- You can set the desired path for the checkpoint saving by changing the checkpoint_dir variable and for the plot saving by changing the `plot_path` variable.
- The environment can be changed by updating the parameter of the `make_env` while initializing `env` variable in `main.py`. Default is the Pong environment.
- You can train the model using the DuelingDeepQNetwork or DuelingDoubleDeepQNetwork by changing the `agent` variable initialization as per the need.

# File Structure
- `main.py` - Performs the main work of running the model over the number of games for learning with initializing the agents, calculating different result statistics, and plotting the results.
- `DuelingDeepQNetwork.py` - Contains the complete architecture of converting the input frame (in form of an array) to getting the outputs (value and advantage functions). It contains the 3 convolution layers (with ReLU function applied) followed by two fully connected (with ReLU) outputing the value and advantage. It also has the optimizer, and loss function.
- `DuelingDQNAgent.py` - This file provides agent for the DuelingDeepQNetwork containing the main learning function. With this, it also contains methods for changing epsilon, getting samples, storing experiences, choosing actions (using epsilon-greedy approach), replacing target networks, etc.
- `DuelingDDQNAgent.py` - This file provides agent for the DuelingDoubleDeepQNetwork having major things similar to the DuelingDQNAgent with some changes in the learn function.
- `ExperienceReplayMemory.py` - This file contains the past experiences observed by the agent while playing the games which becomes useful in learning. It contains the methods for adding an experience, and getting any random experience.
- `utils.py` - This file contains the code for building the environment of the Atari game with methods for preprocessing frames, stacking frames, etc. needed to make it similar to the DeepMind paper.
- `README.md` - This file which gives complete overview of the implementation.
- `requirements.txt` - Provides ready to install all the required libraries for running the implementation.

# Architecture
<p align="center"><img src="https://raw.githubusercontent.com/hemilpanchiwala/Dueling-Network-Architectures/master/images/DuelingDQNArchitecture.png" alt="Architecture"/></p>

The architecture of the Dueling Deep Q Network is quite simple with just one important thing of taking two outputs from the same neural network (instead of one). Here, basically the input_dimensions are passed to the network which are convoluted using three convolution layers with each following the ReLU (Rectified Linear Unit) activation. The first convolution layer convolutes the input to 32 output channels with kernel size of 3 x 3 and stride of 4. The second convolution convolutes the output from the first one into 64 output channels with kernel size of 4 x 4 and stride of 2. The final convolution layer convolutes the 64 channels from the second one to 64 output channels again but with a kerner size of 3 x 3 and stride of 1.

This convoluted outputs are then flattened and passed into the fully connected layers. The first layer applies linear transformation from the flattened outputs dimensions to 1024 which is again linearly transformed to 512 by the next layer. This 512 neurons are then linearly transformed into two outputs separately: value (from 512 to 1) and advantage (from 512 to the number of actions) function.

These value and advantage functions are then used to calculate the Q value of the learning function basically described as below
```
Q_value = Value + (Advantage - mean(Advantage))
```
# Observations
The DuelingDeepQNetwork as well as DuelingDoubleDeepQNetwork agents were trained for 1000 games with storing the scores, epsilon, and steps count. The hyperparameters which provided good results after training are as follows:

| | |
|--|--|
|Learning Rate| 0.0001 |
| Epsilon (at start) | 0.6 |
| Gamma (discount factor) | 0.99 |
| Dec epsilon | 1e-5 |
| Min epsilon | 0.1 |
| Batch size | 32 |
| Replay Memory Size | 1000 |
| Network replace count | 1000 |
| Dec epsilon | 1e-5 |
| Input dimensions | Shape of observation space |
| n_actions | Taken from action space of environment |
| | |

The results of the DuelingDoubleDeepQNetwork agent achieved winning results faster compared to the DuelingDeepQNetwork agent while playing the Pong game ('PongFrameskip-v4'). 

# References
- [Intro to Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) by David Silver (greatly helped to get insights of Reinforcement learning)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) paper
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) paper - for the Atari environment
- Medium Blog - https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
