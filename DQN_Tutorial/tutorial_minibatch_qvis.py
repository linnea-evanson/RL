# Import some modules from other libraries
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import collections
# Import the environment module
from environment import Environment
from q_value_visualiser import QValueVisualiser

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        # Choose randomly between [0 1 2 3], which action to take
        return np.random.choice(4)


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):      #these need to match q_value_visualiser to the in the same order for each state.
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move up only
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2:
            # Move left only
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # Move down only
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


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

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
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

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition_mini_batch):
        #From transition get actual measured reward for a state, and use torch.gather to get the network's predicted reward.
        input = torch.tensor(np.stack(transition_mini_batch[:,0])) #need unsqueeze for single data points, not minibatches. Input is the x,y position #should this be for state or next state?
        actions = torch.tensor(np.stack(transition_mini_batch[:,1]), dtype= torch.int64)
        prediction = self.q_network(input)   #prediction is the Q value for each of the 4 actions. given the input state "input".

        Q_pred = prediction.gather(dim =1, index = actions.unsqueeze(-1)).squeeze(-1)
        reward = torch.tensor(np.stack(transition_mini_batch[:,2])).type(torch.FloatTensor) #, requires_grad= True)   #needed the requires_grad = True for the loss.backward() to work

        # Compute the loss between Q network's prediction, and the actual reward for this state and action
        loss = torch.nn.MSELoss()(Q_pred, reward)   #maybe we should use torch.gather rather than making new variables
        return loss

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen = 5000)
    def add_to_buffer(self, transition):
        self.buffer.append(transition)

    def sample_minbatch(self):
        minibatch_indices = np.random.choice(range(100), 100)  #choose 100 transitions randomly from the buffer
        mini_batch = np.array([self.buffer[i] for i in minibatch_indices], dtype = object)
        return mini_batch

# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    #Create an experience replay buffer
    buffer = ReplayBuffer()

    EPISODE_LENGTH = 20
    TRAINING_ITERATIONS = 100

    #Initialise buffer with 100 transitions
    count = 0
    while count < 101:
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(EPISODE_LENGTH):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            buffer.add_to_buffer(transition)
            count += 1

    # Create lists to store the losses and epochs
    losses = []
    iterations = []

    # Loop over episodes
    for training_iteration in range(TRAINING_ITERATIONS):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(EPISODE_LENGTH):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            buffer.add_to_buffer(transition)

            #Only start training after 100 transitions in the buffer
            mini_batch = buffer.sample_minbatch()
            loss = dqn.train_q_network(mini_batch)


        losses.append(loss) #append final loss for that episode
        training_iteration += 1
        print("Iteration:", training_iteration)
        iterations.append(training_iteration)

    # GET FINAL Q VALUES FOR EACH STATE (EXTRACT FINAL LAYER FROM NETWORK, WITH EXISTING WEIGHTS I THINK?)
    dqn.q_network.eval() #put into eval mode so it doesn't keep training.


    #PROBABLY THIS RESHAPE ISN'T CORRECT, BUT SOMEHOW NEED TO WORK OUT HOW TO CORRECTLY RESHAPE FROM 4X100 TO 10X10X4.
    #Since we are in continuous space we need to create a grid of states that we want to visualise and pass these through the network to get their q values:
    q_values = np.zeros((10,10,4))
    for row in range(10):
        for col in range(10):
            state_tensor = torch.tensor([0.1 * col + 0.05, 0.1 * row + 0.05])
            q_values[col, row, :] = dqn.q_network(state_tensor).detach().numpy()

    #q_values = torch.flip ( torch.reshape( (dqn.q_network.output_layer.weight.data).t(), (10,10,4)),  [0,1]).numpy()
    print(q_values.shape)
    # Draw final Q pattern
    visualiser = QValueVisualiser(environment=environment, magnification=500, starting_pos = (0.3500, 0.1500) )   #not sure if this is working correctly or not???
    # Draw the image
    visualiser.draw_q_values(q_values)    #expecting  q_values  =  np.random.uniform(0, 1, [10, 10, 4])

    #FIND GREEDY POLICY.
    #Find the greedy policy: Given a starting state, what is the max Q action, then move to that state and take next max Q to move there until end of episode.
    agent.reset()
    states = []
    transition = (agent.step())  # get the starting state
    state = torch.tensor(transition[0])
    states.append(state)
    for i in range(EPISODE_LENGTH - 1):
        Q = dqn.q_network(state).detach()   #get all 4 q values for that state
        max_action = torch.argmax(Q)
        # Convert the discrete action into a continuous action.
        continuous_action = agent._discrete_action_to_continuous(max_action)
        # Take one step in the environment, based on the agent's current state.
        next_state, _ = agent.environment.step(state, continuous_action)
        state = next_state.detach().clone()
        states.append(state)
    print(states)
    environment.draw_greedy_pol(states)


    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Episodes', ylabel='Loss', title='Loss Curve for Torch Example')

    # Plot and save the loss vs iterations graph
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("loss_vs_iterations_DQN_shortepisode.png")