import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy, DCGPolicy, LPPolicy, CBCPolicy, GreedyPolicy2, BestFitPolicy, FirstFitPolicy, WorstFitPolicy, NextFitDecreasingPolicy, ExactLinearPolicy
from time import sleep
import numpy as np
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
    input_prods = []
    input_stocks = []

    products = observation["products"]
    for prod in products:
        if prod["quantity"] > 0:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            new_product = solution.Product(prod_w, prod_h)
            for _ in range(prod["quantity"]):
                input_prods.append(new_product)

    stocks = observation["stocks"]
    def _get_stock_size_(stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    stock_id = 0
    for stock in stocks:
        stock_w, stock_h = _get_stock_size_(stock)
        new_stock = solution.Stock(stock_id, stock_w, stock_h)
        stock_id += 1
        input_stocks.append(new_stock)

    # Place products across stocks
    solution.place_products_across_stocks(input_stocks, input_prods)

    result = []
    for stock_idx, stock in enumerate(input_stocks):
        # Iterate over the placed products in the current stock
        for product in stock.placed_products:
            pos_x, pos_y, prod_width, prod_height = product
            result.append({
                "stock_idx": stock_idx,  # Index of the current stock
                "size": (prod_width, prod_height), 
                "position": (pos_x, pos_y)
            })
    for placement in result:
        action = {
            "stock_idx": placement["stock_idx"],
            "size": placement["size"],
            "position": placement["position"]
        }
        # Perform the step with the action
        observation, reward, done, truncated, info = env.step(action)
        print(f"Step result: {observation}, Reward: {reward}")

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

if __name__ == "__main__":
    

    #greedy_test = GreedyPolicyTest()
    test = GuillotineTest()

    # greedy2_test = GreedyPolicy2Test()

    # random_test = RandomPolicyTest()

    # dcg_test = DCGPolicyTest()

    # lp_test = LPPolicyTest()

    # cbc_test = CBCPolicyTest()

    # bf_test = BestFitPolicyTest()

    # ff_test = FirstFitPolicyTest()

    #wf_test = WorstFitPolicyTest()

    # nfd_test = NextFitDecreasingPolicyTest()

    # el_test = ExactLinearPolicyTest()


env.close()
