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

        #print("Initial weights", self.layer_1.weight)
        #print("Initial weights", self.layer_2.weight)
        #print("Initial weights", self.output_layer.weight)

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

    def train_q_network(self, transition):  #Now we use this to train the target network
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, weights = self._calculate_loss(transition)

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item(), weights

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

        weights = reward + Q_pred_next - Q_pred

        # Compute the loss between Q network's prediction, and the actual reward for this state and action
        loss = torch.nn.MSELoss()(Q_pred, reward + self.gamma*Q_pred_next)   #maybe we should use torch.gather rather than making new variables
        return loss, weights


class ReplayBuffer:
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

        return mini_batch



class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        self.new_episode_length = 5000  #for very difficult mazes
        self.reset_trigger = False
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        # Count number of episodes that have occured
        self.episode_count = 0
        self.new_episode_counter = 0
        self.count100lengthepisodes = 0
        # Store current loss of an episode
        self.loss = 0
        #Store loss experienced at each episode
        self.losses = []

        self.epsilon = EPSILON

        # Initialise the buffer and deep q network:
        self.buffer = ReplayBuffer(MINIBATCH_SIZE)
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
                    #print("Greedy pol didn't work :(")
                    self.greedy_pol_trigger = False  #go back to epsilon greedy training

                    # Put network in train mode again and continue learning
                    self.dqn.q_network.train()
                    self.dqn.target_network.train()
                #else:

                    #print("reset start loc")
                return True
            else:
                return False


        if (self.num_steps_taken % self.episode_length == 0) or (self.start_new_episode == True):
            self.episode_count += 1
            if self.episode_length == 100:
                self.count100lengthepisodes += 1


            #Epsilon decay until greedy policy found. Reset when less than 0.1 if greedy pol not found.
            if ((self.episode_length == 100) and (self.greedy_pol_trigger == False) and (self.count100lengthepisodes > 30)) or (self.reset_trigger == True):
                self.gamma = 0.9999999999999999
                self.reset_trigger = True
                self.new_episode_counter += 1
                self.epsilon = EPSILON * np.exp(- self.new_episode_counter/100)   #slower decay of epsilon in this case as we need longer exploring time

                if self.new_episode_length > 100:
                    self.new_episode_length = int(2000* np.exp(- self.new_episode_counter/100))
                #print("epsilon2:", self.epsilon)
                #print("episode length2:", self.new_episode_length)

            else:
                self.epsilon = EPSILON * np.exp(- self.episode_count/70)   #update value of epsilon after each episode
                if self.episode_length > 100:   #decay episode length down to 200, not further decrease otherwise network finds local maxima that may not be goal
                    self.episode_length = int(1000* np.exp(- self.episode_count/50))

                #print("epsilon:", self.epsilon)
                #print("episode length:", self.episode_length)


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
                        #print("greedy pol reaches goal! (stopping training")
                        self.stop_training = True
                else:  #start evaluating greedy pol from start of new episode
                    #print("Started using greedy pol..")
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
            self.buffer.add_to_buffer(transition, tag = True)

        #4. In all other cases TRAIN
        else:
            # Now you can do something with this transition ... #save it in the replay buffer!!!!!!
            self.buffer.add_to_buffer(transition)

            #Now train network, with a minibatch that may include that new transition

            mini_batch = self.buffer.sample_minbatch(self.num_steps_taken)

            temp_loss, weights = self.dqn.train_q_network(mini_batch)
            self.loss += temp_loss

            self.buffer.set_weights(weights)




    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self.dqn.q_network.eval()
        with torch.no_grad():
            Q = self.dqn.q_network(torch.tensor(state)).detach()  # get all 4 q values for that state
            max_action = torch.argmax(Q)
        cont_action = self._discrete_action_to_continuous(max_action)
        #print("state", state, "  action:", max_action)

        return cont_action

