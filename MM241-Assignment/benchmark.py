# Import library
import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2352475.policy2352475 import Solution as Policy2352475
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

NUM_EPISODES = 1600

# CLASS CONTAINER
POLICIES = ["None", "GuillotineCuttingWithBestFit", "SkylineBinPack"]

def benchmark(myPolicy, num_iter, seed, position, queue, policy_id):
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        # render_mode="human",  # Uncomment this line to enable rendering
    )
    fillrat = []
    trimloss = []
    sol_time = []
    
    # pol = myPolicy(policy_id)
    pol = myPolicy()
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
        fillrat.append(info['filled_ratio'])
        trimloss.append(info['trim_loss'])
        sol_time.append(end - start)
        
        queue.put(1)
    
    env.close()
    return fillrat, trimloss, sol_time


def parallel_benchmark(myPolicy, total_iter, policy_id, num_processes=4):
    seeds = [random.randint(0, 1000000) for _ in range(num_processes)]
    chunk_size = (total_iter + num_processes - 1) // num_processes
    
    with mp.Manager() as manager:
        queue = manager.Queue()
        pool = mp.Pool(processes=num_processes)
        
        args_list = [
            (myPolicy, min(chunk_size, total_iter - i * chunk_size), seeds[i], i, queue, policy_id) 
            for i in range(num_processes)
        ]
        
        results = pool.starmap_async(benchmark, args_list)

        with tqdm(total=total_iter, position=0) as pbar:
            completed = 0
            while completed < total_iter:
                completed += queue.get()
                pbar.update(1)

        pool.close()
        pool.join()
        results = results.get()
    
    all_res, all_trim, all_times = [], [], []
    for res, trim, sol_time in results:
        all_res.extend(res)
        all_trim.extend(trim)
        all_times.extend(sol_time)
    
    return all_res, all_trim, all_times


def performance_comparison(Policy_a, Policy_b):
    res_a = benchmark(Policy_a)
    res_b = benchmark(Policy_b)
    
    avg = sum(res_a) / sum(res_b)
    return avg

if __name__ == "__main__":
    """
    Policy Benchmarking software, computer specifications:
    CPU: Ryzen 7 8845HS (8 cores, 16 threads)
    RAM: 32 GB
    """
    # os.system("cls")
    sys.stdout.flush()
    
    global policy_id
    policy_id = int(sys.argv[1])

    all_res, all_trim, all_times = parallel_benchmark(Policy2352475, NUM_EPISODES, policy_id, 8)
    # all_res, all_times = parallel_benchmark(Policy2352475, NUM_EPISODES, 1025)
    os.system("cls")
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
