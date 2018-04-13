# Double Deep Q Network to solve SpaceInvaders

I implemented a convolutional neural network that repeats the one designed in the paper Paper [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Some configurations include learning rate = 1e-5, minibatch number = 32, gamma = 0.99, initial epsilon = 1, memory capacity = 110,000 and burn-in size= 30,000, etc. The pre-processing processes each 4 frames and conducts grayscale and resizing to convert the original RBG image into 84x84 gray image. The input layer take 84x84x4 images and the output layer provides the Q value given the input. Both experience replay and freezing weight techniques are applied. The target network (old Q network with frozen weights) updates every 10,000 iterations, i.e., every 40,000 frames.

I find it necessary to implement a Target Q Network (as in the double Q learningpaper [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)) to stabilize training of the networks.

The agent can be trained to achieve 500+ scores in testing episodes

To run the training, execute:

```python3 Double_DQN_SpaceInvaders.py --env='SpaceInvaders-v0' --train=1```

To test the trained agent, execute

```python3 Double_DQN_SpaceInvaders.py --env='SpaceInvaders-v0' --render=1 --train=0 --model='save/<model name>'```

For example

```python3 Double_DQN_SpaceInvaders.py --env='SpaceInvaders-v0' --render=1 --train=0 --model='save/SpaceInvaders-v0_394best.h5'```





## Videos for Post-Trained Performance

Post-Trained SpaceInvaders

[![SI](https://img.youtube.com/vi/D-txu9_PCTk/0.jpg)](https://youtu.be/D-txu9_PCTk "Post Trained SpaceInvaders - Click to Watch!")

