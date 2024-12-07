import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
# from student_submissions.s2210xxx.policy2352237 import Policy2352237
from student_submissions.s2210xxx.policy2352475 import Solution


# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

def GuillotineTest():
    observation, info = env.reset(seed=42)
    solution = Solution()
    # Prepare input data
    ep =0
    while ep < NUM_EPISODES:
        # Perform the step with the action
        action = solution.get_action(observation, info)
        #print(action)
        # print(info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            #print(info)
            ep += 1

if __name__ == "__main__":
    # Reset the environment
    test = GuillotineTest()
env.close()

