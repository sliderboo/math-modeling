import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random
import json

def generate_stable_stocks():
    """
    Generate a stable set of stocks.
    """
    random.seed(42)  # Ensure reproducibility
    stocks = [(random.randint(20, 30), random.randint(20, 30), 1) for _ in range(5)]
    return stocks

def generate_random_prods(num_prods):
    """
    Generate random products based on the given number of products.
    """
    prods = [(random.randint(5, 15), random.randint(5, 15), 1) for _ in range(num_prods)]
    return prods

def save_input_data(stocks, prods, output_folder):
    """
    Save the stocks and products to an input file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_file = os.path.join(output_folder, "input.txt")
    with open(input_file, "w") as f:
        json.dump({"init_stocks": stocks, "init_prods": prods}, f)
    return input_file

def run_solver(num_stocks, input_file, output_folder):
    """
    Run the exact.py solver with the specified number of stocks and products from an input file.
    """
    command = [
        "python", "exact.py",
        "-f", input_file,
        "-m", str(num_stocks),
        "-o", output_folder
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def parse_solve_time(output):
    """
    Parse the solve time from the solver's output.
    """
    for line in output.splitlines():
        if "Solve time" in line:
            try:
                return float(line.split(":")[1].strip().replace("s", ""))
            except (ValueError, IndexError):
                return None
    return None

def evaluate_performance():
    num_stocks_list = [3, 4, 5]
    num_prods_range = range(1, 11)
    num_trials = 10
    results = {num_stocks: [] for num_stocks in num_stocks_list}

    # Generate stable stocks
    stable_stocks = generate_stable_stocks()
    print(f"Stable stocks: {stable_stocks}")

    for num_prods in num_prods_range:
        print(f"\nEvaluating for {num_prods} products...")
        
        for num_stocks in num_stocks_list:
            print(f"  Number of stocks: {num_stocks}")
            solve_times = []
            
            for i in range(num_trials):
                output_folder = f"temp_output_{num_stocks}_{num_prods}_{i}"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # Generate random products for each trial
                random_prods = generate_random_prods(num_prods)
                input_file = save_input_data(stable_stocks, random_prods, output_folder)
                
                # Run the solver and record solve time
                output = run_solver(num_stocks, input_file, output_folder)
                solve_time = parse_solve_time(output)
                
                if solve_time is not None:
                    solve_times.append(solve_time)
                
                # Clean up temporary output folder
                shutil.rmtree(output_folder, ignore_errors=True)
            
            avg_solve_time = np.mean(solve_times) if solve_times else None
            print(avg_solve_time)
            results[num_stocks].append(avg_solve_time)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for num_stocks in num_stocks_list:
        plt.plot(num_prods_range, results[num_stocks], marker='o', label=f'{num_stocks} Stocks')

    plt.xlabel('Number of Products')
    plt.ylabel('Average Solve Time (s)')
    plt.title('2D Cutting Stock Problem - Performance Evaluation')
    plt.legend()
    plt.grid(True)
    plt.xticks(num_prods_range)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('performance_evaluation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate_performance()
