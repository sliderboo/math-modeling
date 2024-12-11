# Import library
import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2352475.policy2352475 import Policy2352475
from student_submissions.s2352237.policy2352237 import Policy2352237
# Testing library
import random
import multiprocessing as mp
import numpy as np
import time
from tqdm import tqdm
import os

import sys

# orr_seeds 
random.seed(100000)

NUM_EPISODES = 100

# CLASS CONTAINER
POLICIES = ["None", "GuillotineCuttingWithBestFit", "SkylineBinPack"]

def benchmark(myPolicy, num_iter, seed, position, queue, policy_id):
    env = gym.make("gym_cutting_stock/CuttingStock-v0")
    fillrat = []
    trimloss = []
    sol_time = []
    
    pol = myPolicy()  # Initialize the policy
    random.seed(seed)
    this_seeds = [random.randint(0, 1000000) for _ in range(num_iter)]
    
    for ep in range(num_iter):
        observation, info = env.reset(seed=this_seeds[ep])
        terminated = False
        start = time.time()
        while not terminated:
            action = pol.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        
        # Collect the results
        fillrat.append(info['filled_ratio'])
        trimloss.append(info['trim_loss'])
        sol_time.append(end - start)
        queue.put(1)  # Mark the completion of an iteration
    
    env.close()
    return fillrat, trimloss, sol_time

def parallel_benchmark(myPolicy, total_iter, policy_id, num_processes=4):
    seeds = [random.randint(0, 1000000) for _ in range(num_processes)]
    chunk_size = (total_iter + num_processes - 1) // num_processes
    
    with mp.Manager() as manager:
        queue = manager.Queue()
        pool = mp.Pool(processes=num_processes)
        
        # Prepare arguments for each process
        args_list = [
            (myPolicy, min(chunk_size, total_iter - i * chunk_size), seeds[i], i, queue, policy_id)
            for i in range(num_processes)
        ]
        
        # Start parallel execution
        results = pool.starmap_async(benchmark, args_list)
        
        # Track progress without printing from individual processes
        with tqdm(total=total_iter, position=0, desc="Progress") as pbar:
            completed = 0
            while completed < total_iter:
                completed += queue.get()
                pbar.update(1)

        pool.close()
        pool.join()
        
        # Gather results from all processes
        results = results.get()
    
    # Consolidate results
    all_res, all_trim, all_times = [], [], []
    for res, trim, sol_time in results:
        all_res.extend(res)
        all_trim.extend(trim)
        all_times.extend(sol_time)
    
    return all_res, all_trim, all_times


def performance_comparison(Policy_a, Policy_b):
    res_a = benchmark(Policy_a)
    res_b = benchmark(Policy_b)
    
    print(f"Fill ratio comparison: {res_a[0]} vs {res_b[0]}")
    print(f"Trim loss comparison: {res_a[1]} vs {res_b[1]}")
    print(f"Solution time comparison: {res_a[2]} vs {res_b[2]}")

if __name__ == "__main__":
    sys.stdout.flush()  # Ensure immediate flushing of stdout
    global policy_id
    policy_id = int(sys.argv[1])  # Get policy ID from command line
    # Call parallel benchmark and get results
    all_res, all_trim, all_times = parallel_benchmark(Policy2352475, NUM_EPISODES, policy_id, 16)

    # Clear terminal screen (optional)
    os.system("cls")
    
    # Print the final performance evaluation result only once after completion
    print('\n\n\n\n===== Performance evaluation result =====')
    print(f'class: {POLICIES[policy_id]}\n')
    print('Fill ratio')
    print(f'Mean\t{np.mean(all_res) * 100:.2f}%')
    print(f'Min\t{np.amin(all_res) * 100:.2f}%')
    print(f'Max\t{np.amax(all_res) * 100:.2f}%')
    print('Trim loss')
    print(f'Mean\t{np.mean(all_trim) * 100:.2f}%')
    print(f'Min\t{np.amin(all_trim) * 100:.2f}%')
    print(f'Max\t{np.amax(all_trim) * 100:.2f}%')
    print('Solution time')
    print(f'Mean\t{np.mean(all_times):.4f} (s)')
    print(f'Min\t{np.amin(all_times):.4f} (s)')
    print(f'Max\t{np.amax(all_times):.4f} (s)')


    all_res, all_trim, all_times = parallel_benchmark(Policy2352237, NUM_EPISODES, policy_id, 16)

    # Clear terminal screen (optional)
    # os.system("cls")
    
    # Print the final performance evaluation result only once after completion
    print('\n\n\n\n===== Performance evaluation result =====')
    print(f'class: {POLICIES[policy_id]}\n')
    print('Fill ratio')
    print(f'Mean\t{np.mean(all_res) * 100:.2f}%')
    print(f'Min\t{np.amin(all_res) * 100:.2f}%')
    print(f'Max\t{np.amax(all_res) * 100:.2f}%')
    print('Trim loss')
    print(f'Mean\t{np.mean(all_trim) * 100:.2f}%')
    print(f'Min\t{np.amin(all_trim) * 100:.2f}%')
    print(f'Max\t{np.amax(all_trim) * 100:.2f}%')
    print('Solution time')
    print(f'Mean\t{np.mean(all_times):.4f} (s)')
    print(f'Min\t{np.amin(all_times):.4f} (s)')
    print(f'Max\t{np.amax(all_times):.4f} (s)')