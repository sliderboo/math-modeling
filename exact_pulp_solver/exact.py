import pulp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import argparse
import sys

class CSPExactSolver:
    def __init__(self, stocks, prods, output_folder):
        self.init_stocks = stocks
        self.init_prods = prods
        self.output_folder = output_folder

        def write_input_data(self):
            input_data = {
                "init_stocks": self.init_stocks,
                "init_prods": self.init_prods
            }
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            input_file_path = os.path.join(self.output_folder, "input.txt")
            with open(input_file_path, "w") as f:
                f.write(str(input_data))
        
        write_input_data(self)

    def visualize(self, prods, stocks, x, y, s, material, m):
        fig, axes = plt.subplots(m, 1, figsize=(5, m * 7))  # Adjust size for vertical layout
        if m == 1:
            axes = [axes]  # Ensure we can iterate over axes

        i = 0
        colors = []
        for j in range(len(prods)):
            color = "#%06x" % random.randint(0, 0xFFFFFF)
            if i < len(self.init_prods):
                count = self.init_prods[i][2]
                for k in range(count):
                    colors.append(color)
                    j += 1
                i += 1

        for k, ax in enumerate(axes):
            ax.set_xlim(0, stocks[k][0])
            ax.set_ylim(0, stocks[k][1])
            ax.set_title(f"Material {k}", fontsize=16)
            ax.set_xlabel("Width", fontsize=12)
            ax.set_ylabel("Height", fontsize=12)
            ax.set_aspect('equal', adjustable='box')  # Ensure correct scale
            ax.grid(True, linestyle='--', alpha=0.5)

            for i in range(len(prods)):
                if pulp.value(material[i][k]) == 1:  # Check if the item is placed on this material
                    w = prods[i][0] if pulp.value(s[i]) == 1 else prods[i][1]
                    h = prods[i][1] if pulp.value(s[i]) == 1 else prods[i][0]
                    rect = patches.Rectangle((pulp.value(x[i]), pulp.value(y[i])), w, h, linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(
                        pulp.value(x[i]) + w / 2,
                        pulp.value(y[i]) + h / 2,
                        f"{i}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black"
                    )

        plt.tight_layout()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        image_filename = f"{self.output_folder}/visualization.png"
        plt.savefig(image_filename, dpi=300)  # Save the figure as an image file
        # plt.show()

    def solve(self):
        prob = pulp.LpProblem("2D_Cutting_Stock_Problem", pulp.LpMinimize)

        stocks = [(stock[0], stock[1]) for stock in self.init_stocks for _ in range(stock[2])]
        prods = [(w, h) for w, h, d in self.init_prods for _ in range(d)]

        total_prods_area = sum(w * h for w, h in prods)
        stocks_area = stocks[0][0] * stocks[0][1] 
        least_required_stocks = np.ceil(total_prods_area / stocks_area)
        print(f"Total demand area: {total_prods_area}")
        print(f"Total stock area: {stocks_area}")
        print(f"Least required stocks: {least_required_stocks}")

        n = len(prods)  # number of rectangular items
        p = [prod[0] for prod in prods]  # widths of the items
        q = [prod[1] for prod in prods]  # heights of the items
        m = len(stocks)  # number of available materials
        X = [stock[0] for stock in stocks]  # widths of the materials
        X_max = max(X)
        Y = [stock[1] for stock in stocks]  # heights of the materials
        Y_max = max(Y)

        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, cat='Continuous') for i in range(n)]
        s = [pulp.LpVariable(f"s_{i}", cat='Binary') for i in range(n)]
        material = [[pulp.LpVariable(f"material_{i}_{k}", cat='Binary') for k in range(m)] for i in range(n)]

        used_materials = [pulp.LpVariable(f"used_material_{k}", cat='Binary') for k in range(m)]
        prob += pulp.lpSum([used_materials[k] * X[k] * Y[k] for k in range(m)])

        for i in range(n):
            prob += pulp.lpSum([material[i][k] for k in range(m)]) == 1, f"Material_assignment_{i}"
            for k in range(m):
                prob += material[i][k] <= used_materials[k], f"Link_used_material_{i}_{k}"
                prob += x[i] + p[i] * s[i] + q[i] * (1 - s[i]) <= X[k] + X_max * (1 - material[i][k]), f"Bound_X_item_{i}_material_{k}"
                prob += y[i] + q[i] * s[i] + p[i] * (1 - s[i]) <= Y[k] + Y_max * (1 - material[i][k]), f"Bound_Y_item_{i}_material_{k}"

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(m):
                    a_ij = pulp.LpVariable(f"a_{i}_{j}_{k}", cat='Binary')
                    b_ij = pulp.LpVariable(f"b_{i}_{j}_{k}", cat='Binary')
                    c_ij = pulp.LpVariable(f"c_{i}_{j}_{k}", cat='Binary')
                    d_ij = pulp.LpVariable(f"d_{i}_{j}_{k}", cat='Binary')
                    prob += a_ij + b_ij + c_ij + d_ij == 1, f"Nonoverlap_condition_{i}_{j}_material_{k}"

                    prob += x[i] + p[i] * s[i] + q[i] * (1 - s[i]) <= x[j] + X[k] * (1 - a_ij + 1 - material[i][k] + 1 - material[j][k]), f"Nonoverlap_a_{i}_{j}_material_{k}"
                    prob += x[j] + p[j] * s[j] + q[j] * (1 - s[j]) <= x[i] + X[k] * (1 - b_ij + 1 - material[i][k] + 1 - material[j][k]), f"Nonoverlap_b_{i}_{j}_material_{k}"
                    prob += y[i] + q[i] * s[i] + p[i] * (1 - s[i]) <= y[j] + Y[k] * (1 - c_ij + 1 - material[i][k] + 1 - material[j][k]), f"Nonoverlap_c_{i}_{j}_material_{k}"
                    prob += y[j] + q[j] * s[j] + p[j] * (1 - s[j]) <= y[i] + Y[k] * (1 - d_ij + 1 - material[i][k] + 1 - material[j][k]), f"Nonoverlap_d_{i}_{j}_material_{k}"

        prob.solve()

        if pulp.LpStatus[prob.status] == 'Optimal':  # Optimal solution found
            used_materials = [0] * m
            total_used_area = [0] * m

            for i in range(n):
                w = prods[i][0] if pulp.value(s[i]) == 1 else prods[i][1]
                h = prods[i][1] if pulp.value(s[i]) == 1 else prods[i][0]
                assigned_material = [k for k in range(m) if pulp.value(material[i][k]) == 1][0]
                used_materials[assigned_material] = 1  # Mark this material as used
                total_used_area[assigned_material] += w * h

            total_materials_used = sum(used_materials)
            total_waste = 0
            for k in range(m):
                if used_materials[k] == 1:
                    total_material_area = X[k] * Y[k]
                    total_waste += total_material_area - total_used_area[k]
                    print(f"""[+] Material {k}: 
        [-] Total area = {total_material_area}
        [-] Used area = {total_used_area[k]}
        [-] Waste area = {X[k] * Y[k] - total_used_area[k]}
        [-] Fill percentage: {round(total_used_area[k] / (total_used_area[k] + X[k] * Y[k] - total_used_area[k]) * 100, 2)}%""")

            # Visualization
            self.visualize(prods, stocks, x, y, s, material, m)
            return
    
        else: 
            print(f"Solver ended with status {pulp.LpStatus[prob.status]}")
            return


def generate_data(n, m):
    stocks = [(random.randint(20, 30), random.randint(20, 30), random.randint(1, 1)) for _ in range(m)]
    prods = [(random.randint(5, 15), random.randint(5, 15), random.randint(1, 1)) for _ in range(n)]
    return stocks, prods

def read_input_file(filename):
    if not os.path.exists(filename):
        sys.exit(f"File {filename} not found.")
    with open(filename, "r") as f:
        data = eval(f.read())
    return data["init_stocks"], data["init_prods"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Cutting Stock Problem Solver")
    parser.add_argument("-r", action="store_true", help="Generate random data")
    parser.add_argument("-m", type=int, default=3, help="Number of stocks for random generation")
    parser.add_argument("-n", type=int, default=6, help="Number of products for random generation")
    parser.add_argument("-f", type=str, help="Input file containing stocks and products")
    parser.add_argument("-o", type=str, default="output", help="Output folder to store results")

    args = parser.parse_args()

    init_stocks = []
    init_prods = []

    if args.r and args.m and args.n:
        init_stocks, init_prods = generate_data(args.n, args.m)
    elif args.f:
        init_stocks, init_prods = read_input_file(args.f)
    else:
        sys.exit("Invalid arguments. Use -r with -m and -n for random data or -f to specify an input file.")

    solver = CSPExactSolver(init_stocks, init_prods, args.o)
    solver.solve()
