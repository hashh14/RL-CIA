import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.7
LR = 0.5
LOOP_PENALTY = -5

np.random.seed(56)

class Experience:
    def __init__(self):
        self.current = None
        self.replay = []
        self.actions = np.concatenate([np.eye(2),-1*np.eye(2)], axis=0)
    
    def start_new_episode(self):
        if self.current is not None:
            self.replay.append(self.current)
        
        self.current = None
        
    def store_set(self,SAR:dict):
        add = np.array(list(SAR.values())).reshape(1,-1)
        if self.current is not None:
            self.current = np.concatenate([self.current,add],axis=0)
        else:
            self.current = add
        
        return 1
    
    def get_latest_reward(self):
        total = np.sum(self.current[:,-1])
        return total
    
    def get_path(self):
        for path in self.replay:
            turns = path.shape[0]
            total_reward = np.sum(path[:,-1])
            if turns >= total_reward:
                yield path
    
def check_loop(exp:Experience, next_state, reward):
    if exp.current is not None:
        for ex in exp.current:
            if ex[0] == next_state and ex[-1]>0:
                print("LOOP FOUND - ",ex, next_state, reward)
                return 1
    return 0

def generate_grid(size=50, num_obstacles=100):
    grid = np.zeros((size,size))
    coord = np.random.randint(0,size,(num_obstacles,2))
    for obs in coord:
        grid[obs[0],obs[1]] = 1
    
    ## Goal cannot be an obstacle
    grid[0,size-1] = 0
    plt.imshow(grid, cmap='gray')
    plt.show()
    return grid

def epsilon_greedy(q_value, prob=0.8, current_action=True):
    rand = np.random.rand()
    if prob<rand:
        if current_action:
            value = np.argmax(q_value)
        else:
            value = np.max(q_value)
    else:
        if current_action:
            value = np.random.choice(np.arange(q_value.size))
        else:
            value = np.random.choice(q_value)
            
    return value
    
def get_reward(state_action: dict, maze, grid_size):
    next = state_action["state"] + state_action["action"]
    print(state_action["state"], next)
    
    if np.any(next < [0,0]) or np.any(next > [grid_size-1, grid_size-1]):
        return -10, "failed"
    elif maze[next[0],next[1]]:
        return -10, "obstacle"
    elif np.all(next == [0, grid_size-1]):
        return 10, "finished"
    else:
        return 1, "success"
    
def episode(maze, current_position, experience_buffer: Experience, Q_value, grid_size):
    ACTIONS = np.concatenate([np.eye(2,dtype=np.int8),-1*np.eye(2,dtype=np.int8)], axis=0, dtype=np.int8)
    
    for i in range(5000):
        current_action = epsilon_greedy(Q_value[current_position[0],current_position[1],:])
        print(f"ACTIONS TAKEN - {ACTIONS[current_action]}")
        x,y = current_position+ACTIONS[current_action]
        
        current_reward, flag = get_reward(
            {"state":current_position, "action":ACTIONS[current_action]}, 
            maze=maze, 
            grid_size=grid_size
        )
        
        if check_loop(experience_buffer, grid_size*x + y, current_reward):
            current_reward = LOOP_PENALTY
            flag = "failed"
        
        print(f"REWARD RECEIVED - {current_reward}")
        
        if flag == "finished":
            return 1
        elif flag == "failed":
            position_int = grid_size*current_position[0] + current_position[1]
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
            next_state_Q = 0
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
            break
        elif flag == "obstacle":
            x,y = None, None
            position_int = grid_size*current_position[0] + current_position[1]
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
        else:
            position_int = grid_size*current_position[0] + current_position[1]
            experience_buffer.store_set({"state":position_int, "action": current_action, "reward":current_reward})
        
        if x is None and y is None:
            next_state_Q = 0
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
        else:
            next_state_Q = epsilon_greedy(Q_value[x,y,:], current_action=False)
            current_Q = Q_value[current_position[0],current_position[1],current_action]
            current_Q = current_Q + LR*(current_reward + GAMMA*next_state_Q - current_Q)
            Q_value[current_position[0],current_position[1],current_action] = current_Q
            print("Q-VALUES - ", Q_value[current_position[0],current_position[1],:])
            current_position = np.array([x,y])
            
        print("POS - ",current_position,end="\n")
        if i%1000 == 0:
            print(f"ITERATION - {i}\n")
    
    return

def save_to_file(grid_size, num_obstacles, total_episodes, total_reward, Q_value):
    filename = f"navigation_results_SARSA.txt"
    
    with open(filename, 'w') as f:
        f.write("Grid Navigation Results\n")
        f.write("=====================\n\n")
        f.write(f"Grid Size: {grid_size}x{grid_size}\n")
        f.write(f"Number of Obstacles: {num_obstacles}\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Final Total Reward: {total_reward}\n\n")
        f.write("Final Q-Values:\n")
        f.write("=============\n")
        for i in range(grid_size):
            for j in range(grid_size):
                f.write(f"\nPosition ({i},{j}):\n")
                f.write(f"Right: {Q_value[i,j,0]:.2f}\n")
                f.write(f"Up: {Q_value[i,j,1]:.2f}\n")
                f.write(f"Left: {Q_value[i,j,2]:.2f}\n")
                f.write(f"Down: {Q_value[i,j,3]:.2f}\n")
    
    print(f"\nResults have been saved to {filename}")

def main():
    grid_size = int(input("Enter grid size: "))
    obs = int(input("Enter the number of obstructions: "))
    
    # Generate the grid
    maze = generate_grid(grid_size, obs)
    current_agent_location = np.array([grid_size-1,0], dtype=np.int8)
    experience = Experience()
    Q_value = np.abs(np.random.normal(0,5,(grid_size,grid_size,4)))
    num_episode = 1
    
    while(True):
        status = episode(maze, current_agent_location, experience, Q_value, grid_size)
        if status:
            print("The destination has been reached")
            total_reward = experience.get_latest_reward()
            # Save results to file
            save_to_file(grid_size, obs, num_episode, total_reward, Q_value)
            break
        total_reward = experience.get_latest_reward()
        print(f"EPISODE {num_episode} REWARD - ", total_reward, end="\n\n")
        num_episode += 1
        experience.start_new_episode()

if __name__ == "__main__":
    main()