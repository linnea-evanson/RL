############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections
import random
import time

start_time = time.time()    #use training time to set decay rates of epsilon and episode length
end_time = start_time + 600


#Hyperparameters:
TARGET_UPDATE_FREQ = 10  #update after this number of EPISODES (not steps of an episode)
MINIBATCH_SIZE = 500
LEARNING_RATE = 0.001
GAMMA = 0.999
EPSILON = 1 #make epsilon decay with number of episodes.


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()  #calls innit from torch.nn.Module
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

        print("Initial weights", self.layer_1.weight)
        print("Initial weights", self.layer_2.weight)
        print("Initial weights", self.output_layer.weight)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, gamma, learning_rate):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=6)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr= learning_rate)

        # Create a target network to stabilise training
        self.target_network = Network(input_dimension=2, output_dimension=6)   #doesn't have an optimiser, is only updated by copying weights from q_network

        #Set gamma:
        self.gamma = gamma
        self.error = 0 # for prioritised replay buffer


    def train_q_network(self, transition):  #Now we use this to train the target network
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def update_Q_from_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict()) #only loaded when this is called. Not constantly updated as they are not set equal but loaded.


    # Function to calculate the loss for a particular transition.
    #!!!  Q values should be computed from the target network, but then the q_network should be optimised from that. -add .detach() each time target network is called.
    def _calculate_loss(self, transition_mini_batch):
        #Use Bellman's equation to find expected discounted sum of future rewards and compare that to the measured reward?

        #Current state (calculated by q_networks
        s = torch.tensor(np.stack(transition_mini_batch[:,0]), dtype = torch.float)#current state
        actions = torch.tensor(np.stack(transition_mini_batch[:,1]), dtype= torch.int64)
        prediction = self.q_network(s)   #prediction is the Q value for the state and each of the 4 actions.
        Q_pred = prediction.gather(dim =1, index = actions.unsqueeze(-1)).squeeze(-1)

        #Next state (calculated by target network as used for ground truth comparison)
        s_prime = torch.tensor(np.stack(transition_mini_batch[:, 3]), dtype = torch.float)  # next state
        prediction2 = self.target_network(s_prime.detach()).detach()  # prediction is the Q value for the state and each of the 4 actions.
        max_actions = torch.argmax(prediction2, dim = 1)
        Q_pred_next = prediction2.gather(dim=1, index=max_actions.unsqueeze(-1)).squeeze(-1)    #index needs to be the max in that row

        reward = torch.tensor(np.stack(transition_mini_batch[:,2])).type(torch.FloatTensor)

        self.error = reward + Q_pred_next - Q_pred

        # Compute the loss between Q network's prediction, and the actual reward for this state and action
        loss = torch.nn.MSELoss()(Q_pred, reward + self.gamma*Q_pred_next)   #maybe we should use torch.gather rather than making new variables
        return loss

class SumTree(object):
    pointer = 0

    def __init__(self, maxsize):
        self. maxsize = maxsize
        self.tree = np.zeros(2* maxsize -1)
        self.data = np.zeros(maxsize, dtype = object)
        self.num_filled = 0

    """def propagate(self, idx, delta):
        parent = (idx -1) // 2    #integer division
        self.tree[parent] += delta

        if not parent == 0:
            self.propagate(parent, delta)  #continue through all parent nodes until last one which is zero"""

    def get_sample(self, s):
        idx = 0 #start at very top node (we search downwards from there)
        leaf_idx = 0

        while True:
            left = 2*idx + 1  #left below idx (so idx is the parent of left and right)
            right = left + 1   #right below idx

            if left >= len(self.tree): #there are no more leaves to the left below of idx so end search (idx doesn't have kids).
                leaf_idx = idx
                break
            else:
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = right
        data_idx = leaf_idx - self.maxsize + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def add_entry(self, priority, data):
        idx = self.pointer + self.maxsize - 1 #start at other side of the tree
        self.data[self.pointer] = data
        self.update(idx, priority)

        self.pointer += 1

        if self.pointer >= self.maxsize:  #replace when tree is full
            self.pointer = 0
        #if self.num_filled < self.maxsize:
        #    self.num_filled += 1

    def update(self, idx, priority):
        delta = priority - self.tree[idx]   #the amount we are going to change the priority of this node

        self.tree[idx] = priority
        #self.add_entry(idx, delta)
        while idx != 0:
            idx = (idx - 1) //2
            self.tree[idx] += delta


    """def get_node(self, s):
        idx = self.get_sample(0,s) #get sample at 0 because we will really check through the whole tree
        idx2 = idx - self.maxsize + 1
        return (idx, self.tree[idx], self.data[idx2])"""


class Prioritised_ReplayBuffer:
    avoid = 0.01
    highprior_vs_rand = 0.6
    importance_sampling = 0.4
    increment = 0.001

    def __init__(self, maxsize):
        self.tree = SumTree(maxsize)
        self.maxsize = maxsize

    def _get_priority(self, error):
        return (np.abs(error) + self.avoid) ** self.highprior_vs_rand

    def add_to_buffer(self, error, sample):
        #p = self._get_priority(error)
        p = np.max(self.tree.tree[self.tree.maxsize:])  # make new transitions highest priority
        if p == 0: #(this is the first entry in the episode)
            p = 1
        self.tree.add_entry(p, sample)

    def sample_minibatch(self, n):
        batch = np.zeros((n, len(self.tree.data[0])), dtype = object)
        idxs = np.zeros((n,))
        segment = self.tree.total() / n
        is_weight = np.zeros((n, 1))

        self.importance_sampling = np.min([1., self.importance_sampling + self.increment])

        min_p = np.min(self.tree.tree[-self.tree.maxsize:])/self.tree.total()
        if min_p == 0:
            min_p = 0.00001
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_sample(s)

            prob = p/self.tree.total()  #normalised probability of this node
            batch[i,0] = data[0]
            batch[i,1] = data[1]
            batch[i,2] = data[2]
            batch[i,3] = data[3]

            idxs[i] = idx
            is_weight[i,0] = np.power(prob/min_p, -self.importance_sampling)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

"""class ReplayBuffer:
    def __init__(self, MINIBATCH_SIZE):
        self.buffer = collections.deque(maxlen = 5000)
        self.minibatch_size = MINIBATCH_SIZE
        self.weights = collections.deque(maxlen = 5000) #for prioritised experience replay

    def add_to_buffer(self, transition, tag = False):
        # If our replay memory is full, elements will be automatically popped from opposite end
        self.buffer.append(transition)

        if tag == True: #now we are making the very first minibatch, before starting training, make it uniform and small weights
            self.weights.append(0.001) #the first weight in the list
        else:
            self.weights.append(max(self.weights))  #when adding a new transition, set it to max of all current weights so it will get trained on in next step.


    def set_weights(self, weights): #all this weights stuff is for prioritised experience replay buffer
        #w = weights.detach()
        #for ind, val in enumerate(self.minibatch_indices):
        #    self.weights[val] = (abs(w[ind]) + 0.00000001)**2
        pass

    def sample_minbatch(self, steps_taken):   #let's do prioritised experience replay
        #self.minibatch_indices = random.choices(range(len(self.buffer)), k = self.minibatch_size, weights = self.weights)  #prioritised weighted choice

        self.minibatch_indices = random.choices(range(len(self.buffer)), k = self.minibatch_size)  #choose 100 transitions randomly from the buffer
        mini_batch = np.array([self.buffer[i] for i in self.minibatch_indices], dtype = object)

        return mini_batch"""



class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        # Count number of episodes that have occured
        self.episode_count = 0
        # Store current loss of an episode
        self.loss = 0
        #Store loss experienced at each episode
        self.losses = []

        self.epsilon = EPSILON

        # Initialise the buffer and deep q network:
        self.buffer = Prioritised_ReplayBuffer(5000) #max size 5000
        self.dqn = DQN(GAMMA, LEARNING_RATE)

        #Trigger greedy policy evaluation
        self.greedy_pol_trigger = False
        self.greedy_pol_steps = 0
        self.stop_training = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.greedy_pol_trigger: #if evaluating greedy pol keep episode going for 100 steps of greedy pol
            if self.start_new_episode: #when we very first start evaluating a greedy policy we need to first reset the starting location
                self.episode_count += 1
                self.losses.append(self.loss)  # save sum of loss for the episode
                self.loss = 0  # set loss for the episode back to zero
                self.start_new_episode = False
                return True #should we also update loss and epsilon etc here?

            if self.greedy_pol_steps == 100:
                self.greedy_pol_steps = 0
                if not self.stop_training:  #if our greedy policy has not brought us to the goal, set networks back in training mode
                    print("Greedy pol didn't work :(")
                    self.greedy_pol_trigger = False  #go back to epsilon greedy training

                    # Put network in train mode again and continue learning
                    self.dqn.q_network.train()
                    self.dqn.target_network.train()
                else:
                    print("reset start loc")
                return True
            else:
                return False


        if (self.num_steps_taken % self.episode_length == 0) or (self.start_new_episode == True):
            self.episode_count += 1
            self.epsilon = EPSILON * np.exp(- self.episode_count/50)   #update value of epsilon after each episode
            if self.episode_length > 100:   #decay episode length down to 200, not further decrease otherwise network finds local maxima that may not be goal
                self.episode_length = int(1000* np.exp(- self.episode_count/50))

            print("epsilon:", self.epsilon)
            print("episode length:", self.episode_length)
            self.losses.append(self.loss)  #save sum of loss for the episode
            self.loss = 0           #set loss for the episode back to zero
            if self.episode_count % TARGET_UPDATE_FREQ == 0: #every certain number of episodes update target network
                self.dqn.update_Q_from_target()

            self.start_new_episode = False #set back to false
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1

        if self.num_steps_taken < MINIBATCH_SIZE: #epsilon = 0 when building buffer
            action = random.choices([0, 1, 2, 3, 4, 5])[0]  # returns list within a list, so we return just the inner list

            # Store the state; this will be used later, when storing the transition
            self.state = state
            # Store the action; this will be used later, when storing the transition
            self.action = action

            cont_action = self._discrete_action_to_continuous(action)
            return cont_action

        #Greedy pol
        if self.greedy_pol_trigger:
            self.greedy_pol_steps += 1 #update number of steps taken (we will continue until this is 100)

            with torch.no_grad():
                Q = self.dqn.q_network(torch.tensor(state, dtype = torch.float)).detach()  # get all 4 q values for that state
                max_action = torch.argmax(Q)
            # Store the state; this will be used later, when storing the transition
            self.state = state
            # Store the action; this will be used later, when storing the transition
            self.action = max_action

            cont_action = self._discrete_action_to_continuous(max_action)


        #Epsilon greedy policy
        else:
            Q = self.dqn.q_network(torch.tensor(state, dtype = torch.float))  # get all 4 q values for that state
            max_action = torch.argmax(Q)

            prob_all = self.epsilon / 6
            weights = [prob_all, prob_all, prob_all, prob_all, prob_all, prob_all]  #set probability of moving left to zero
            weights[max_action] = 1 - self.epsilon + self.epsilon / 6

            action = random.choices([0,1,2,3,4,5], weights=weights)[0]  # returns list within a list, so we return just the inner list

            # Store the state; this will be used later, when storing the transition
            self.state = state
            # Store the action; this will be used later, when storing the transition
            self.action = action

            cont_action = self._discrete_action_to_continuous(action)

        return cont_action

        # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self,discrete_action):  # these need to match q_value_visualiser to the in the same order for each state.
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move up only
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 2:
            # Move left only
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 3:
            # Move down only
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        if discrete_action == 4:
            # Diagonal up right
            continuous_action = np.array([0.01414, 0.01414], dtype=np.float32)
        elif discrete_action == 5:
            # Diagonal down right
            continuous_action = np.array([0.01414, -0.01414], dtype=np.float32)
        """elif discrete_action == 6:
            # Move left only
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 7:
            # Move down only
            continuous_action = np.array([0, -0.02], dtype=np.float32)"""

        return continuous_action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        #1. CALCULATE REWARD WHEN NEAR GOAL---------------
        if distance_to_goal < 0.04: # do not decrease reward for staying in the same place when we are near goal
            reward = float(0.1 * (1 - distance_to_goal))

            #IF REACHED GOAL CONSIDER GREEDY POLICY--------
            #if distance_to_goal <= 0.03 and self.epsilon < 0.03 and self.episode_length == 100:  # Check greedy policy and if satisfactory stop training.
            if distance_to_goal <= 0.03:
                if self.greedy_pol_trigger: #already evaluating greedy pol
                    if distance_to_goal < 0.03:
                        print("greedy pol reaches goal! (stopping training")
                        self.stop_training = True
                else:  #start evaluating greedy pol from start of new episode
                    print("Started using greedy pol..")
                    self.start_new_episode = True
                    self.greedy_pol_trigger = True

                    # Put network in eval mode so it doesn't keep training during this evaluation of its greedy policy
                    self.dqn.q_network.eval()
                    self.dqn.target_network.eval()
                    torch.no_grad()
        #CALCULATE REWARD WHEN FAR FROM GOAL-----------------
        else:
            if all( x == y for x, y in zip(next_state,self.state)):     #prevents getting stuck against a wall or returning to where we were
                reward = 0
            else:
                # Convert the distance to a reward
                reward = float(0.1*(1 - distance_to_goal))* np.exp((-self.num_steps_taken % self.episode_length)/self.episode_length)


        #2. Create a transition with that reward-----------------
        transition = (self.state, self.action, reward, next_state)

        #3. If training making first minibatch, or evaluating greedy pol DONT TRAIN --------------
        if (self.num_steps_taken < MINIBATCH_SIZE) or (self.greedy_pol_trigger == True):   #save the first 100 transitions to be the first minibatch then begin training
            self.buffer.add_to_buffer(self.dqn.error, transition)
        #4. In all other cases TRAIN
        else:
            # Now you can do something with this transition ... #save it in the replay buffer!!!!!!
            self.buffer.add_to_buffer(self.dqn.error, transition)

            #Now train network, with a minibatch that may include that new transition

            mini_batch, _ , _ = self.buffer.sample_minibatch(MINIBATCH_SIZE)

            temp_loss = self.dqn.train_q_network(mini_batch)
            self.loss += temp_loss

            #self.buffer.set_weights(weights)




    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self.dqn.q_network.eval()
        with torch.no_grad():
            Q = self.dqn.q_network(torch.tensor(state)).detach()  # get all 4 q values for that state
            max_action = torch.argmax(Q)
        cont_action = self._discrete_action_to_continuous(max_action)
        print("state", state, "  action:", max_action)

        return cont_action

