import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy
from student_submissions.s2352237.policy2352237 import Policy2352237
from student_submissions.s2352475.policy2352475 import Policy2352475
from student_submissions.s2352216_2352237_2352334_2352475.policy2352216_2352237_2352334_2352475 import Policy2352216_2352237_2352334_2352475
import time

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    # render_mode = "rgb_array",
    # render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 10

def GuillotineTest():
    observation, info = env.reset(seed=42)
    solution = Policy2352475()
    # Prepare input data
    ep =0
    while ep < NUM_EPISODES:
        # Perform the step with the action
        action = solution.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1
    solution.print_performance()

def SkylineTest():
    observation, info = env.reset(seed=42)
    solution = Policy2352237()
    # Prepare input data
    sol_time = []
    ep = 0
    flag = False
    while ep < NUM_EPISODES:
        # Perform the step with the action
        action = solution.get_action(observation, info)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if not flag:
            start = time.time()
            flag = True
        if terminated or truncated:
            end = time.time()
            flag = False
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1
            sol_time.append(end - start)
    solution.print_performance()
    print("Max Time To Solve: ", max(sol_time))
    print("Min Time To Solve: ", min(sol_time))
    print("Avg Time To Solve: ", sum(sol_time) / len(sol_time))
    
def MergeTest():
    # Reset the environment
    observation, info = env.reset(seed=42)

    policy2352216_2352237_2352334_2352475 = Policy2352216_2352237_2352334_2352475(policy_id=2)
    for _ in range(200):
        action = policy2352216_2352237_2352334_2352475.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            observation, info = env.reset()


if __name__ == "__main__":
    # Reset the environment
    # test = GuillotineTest()
    test = SkylineTest()
    # test = MergeTest()
        
env.close()

