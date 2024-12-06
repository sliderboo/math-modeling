import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy, DCGPolicy, LPPolicy, CBCPolicy, GreedyPolicy2, BestFitPolicy, FirstFitPolicy, WorstFitPolicy, NextFitDecreasingPolicy, ExactLinearPolicy, BieuPolicy
from time import sleep
# from student_submissions.s2210xxx.policy2352237 import Policy2352237

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 1

def GreedyPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def RandomPolicyTest():
    observation, info = env.reset(seed=42)

    # Test RandomPolicy
    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

def DCGPolicyTest():
    observation, info = env.reset(seed=42)

    # Test DCGPolicy
    dcg_policy = DCGPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = dcg_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)   

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

def LPPolicyTest():
    observation, info = env.reset(seed=42)

    # Test LPPolicy
    lp_policy = LPPolicy()
    ep = 0

    print ("debug")

    while ep < NUM_EPISODES:
        action = lp_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)   

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

def CBCPolicyTest():
    observation, info = env.reset(seed=42)

    # Test CBCPolicy
    cbc_policy = CBCPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = cbc_policy.get_action(observation, info)
        print(action)
        sleep(0.5)
        if (action == None):
            sleep(300)
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

def GreedyPolicy2Test():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    gd_policy2 = GreedyPolicy2()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy2.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def BestFitPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    bf_policy = BestFitPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = bf_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def FirstFitPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    ff_policy = FirstFitPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = ff_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def WorstFitPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    wf_policy = WorstFitPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = wf_policy.get_action(observation, info)
        # print ([(prod["size"][0], prod["size"][1], prod["quantity"]) for prod in wf_policy.list_prods])
        # print ([wf_policy._get_stock_size_(stock) for stock in observation["stocks"]])
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def NextFitDecreasingPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    nfd_policy = NextFitDecreasingPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = nfd_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def ExactLinearPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    el_policy = ExactLinearPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = el_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

def BieuPolicyTest():
    observation, info = env.reset(seed=42)
    
    # Test GreedyPolicy
    bieu_policy = BieuPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = bieu_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action) 

        if terminated or truncated:
            sleep(300)
            observation, info = env.reset(seed=ep)
            ep += 1

if __name__ == "__main__":

    # greedy_test = GreedyPolicyTest()

    # greedy2_test = GreedyPolicy2Test()

    # random_test = RandomPolicyTest()

    # dcg_test = DCGPolicyTest()

    # lp_test = LPPolicyTest()

    # cbc_test = CBCPolicyTest()

    # bf_test = BestFitPolicyTest()

    # ff_test = FirstFitPolicyTest()

    wf_test = WorstFitPolicyTest()

    # nfd_test = NextFitDecreasingPolicyTest()

    # el_test = ExactLinearPolicyTest()

    # bieu_test = BieuPolicyTest()


env.close()
