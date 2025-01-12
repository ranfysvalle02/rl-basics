import numpy as np  
import random  
  
# Define the maze  
maze = [  
    [0, 0, 0, 0, 0],  # 0 represents a wall  
    [0, 1, 1, 1, 0],  # 1 represents a path  
    [0, 1, 9, 1, 0],  # 9 represents the goal  
    [0, 1, 1, 1, 0],  
    [0, 0, 0, 0, 0]  
]  
  
# Maze dimensions  
maze_height = len(maze)  
maze_width = len(maze[0])  
  
# Actions the agent can take  
actions = ['up', 'down', 'left', 'right']  
  
# Create a Q-table initialized to zeros  
# Dimensions: maze_height x maze_width x number of actions  
q_table = np.zeros((maze_height, maze_width, len(actions)))  
  
# Hyperparameters  
alpha = 0.1      # Learning rate  
gamma = 0.9      # Discount factor  
epsilon = 0.1    # Exploration rate  
episodes = 1000  # Number of times the agent tries to learn  
  
# Starting position  
start_position = (1, 1)  
  
# Function to get the next position based on the action  
def get_next_position(position, action):  
    x, y = position  
    if action == 'up':  
        x -= 1  
    elif action == 'down':  
        x += 1  
    elif action == 'left':  
        y -= 1  
    elif action == 'right':  
        y += 1  
    return (x, y)  
  
print("Starting Q-learning training...")  
  
# Training the agent  
for episode in range(episodes):  
    position = start_position  
    done = False  
  
    while not done:  
        x, y = position  
  
        # Decide whether to explore or exploit  
        if random.uniform(0, 1) < epsilon:  
            # Explore: choose a random action  
            action = random.choice(actions)  
            print(f"Episode {episode}: Exploring action '{action}' at position {position}.")  
        else:  
            # Exploit: choose the best action from Q-table  
            action_index = np.argmax(q_table[x, y])  
            action = actions[action_index]  
            print(f"Episode {episode}: Exploiting action '{action}' at position {position}.")  
  
        # Get the next position based on the action  
        next_position = get_next_position(position, action)  
        x_next, y_next = next_position  
  
        # Check if the next move is within the maze boundaries  
        if x_next < 0 or x_next >= maze_height or y_next < 0 or y_next >= maze_width:  
            reward = -1  # Negative reward for hitting a wall  
            next_position = position  # Stay in the same place  
            print(f"  Hit the boundary! Staying at position {position}.")  
        elif maze[x_next][y_next] == 0:  
            reward = -1  # Negative reward for hitting a wall  
            next_position = position  # Stay in the same place  
            print(f"  Hit a wall at position {next_position}! Staying at position {position}.")  
        elif maze[x_next][y_next] == 9:  
            reward = 10  # Positive reward for reaching the goal  
            done = True  # Goal reached  
            print(f"  Reached the goal at position {next_position}!")  
        else:  
            reward = -0.1  # Small negative reward for each move  
            print(f"  Moved to position {next_position}.")  
  
        # Update Q-table  
        action_index = actions.index(action)  
        old_value = q_table[x, y, action_index]  
        next_max = np.max(q_table[x_next, y_next])  
        # Q-learning formula  
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)  
        q_table[x, y, action_index] = new_value  
  
        # Move to the next position  
        position = next_position  
  
print("\nTraining completed.")  
  
# After training, find the best path  
position = start_position  
done = False  
path = [position]  
  
print("\nFinding the best path based on the learned Q-table...")  
  
while not done:  
    x, y = position  
    # Choose the best action based on the learned Q-table  
    action_index = np.argmax(q_table[x, y])  
    action = actions[action_index]  
    next_position = get_next_position(position, action)  
    x_next, y_next = next_position  
  
    # Check if the next move leads to the goal  
    if maze[x_next][y_next] == 9:  
        path.append(next_position)  
        print(f"Moved '{action}' to {next_position}. Goal reached!")  
        done = True  
    elif maze[x_next][y_next] == 0 or x_next < 0 or x_next >= maze_height or y_next < 0 or y_next >= maze_width:  
        print(f"Encountered a wall or boundary when moving '{action}'. Cannot proceed further.")  
        done = True  
    else:  
        position = next_position  
        path.append(position)  
        print(f"Moved '{action}' to {position}.")  
  
# Print the path taken  
print("\nPath to the goal:")  
for step in path:  
    print(step)  
