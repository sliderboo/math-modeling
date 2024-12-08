import coptpy as cp
from coptpy import COPT
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

    def _get_info(self, cutted_stocks, stocks):
        filled_ratio = []
        trim_loss = []

        for i in range(len(cutted_stocks)):
            filled_ratio.append(cutted_stocks[i])
            trim_loss.append(1 - cutted_stocks[i])

        return {"filled_ratio": filled_ratio, "trim_loss": trim_loss}

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
                if round(material[i][k].x) == 1:  # Check if the item is placed on this material
                    w = prods[i][0] if round(s[i].x) == 1 else prods[i][1]
                    h = prods[i][1] if round(s[i].x) == 1 else prods[i][0]
                    rect = patches.Rectangle((x[i].x, y[i].x), w, h, linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(
                        x[i].x + w / 2,
                        y[i].x + h / 2,
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
        env = cp.Envr()
        prob = env.createModel("2D_Cutting_Stock_Problem")

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

        x = prob.addVars(n, vtype=COPT.CONTINUOUS, nameprefix="x")
        y = prob.addVars(n, vtype=COPT.CONTINUOUS, nameprefix="y")
        s = prob.addVars(n, vtype=COPT.BINARY, nameprefix="s")  
        material = [[prob.addVar(vtype=COPT.BINARY, name=f"material_{i}_{k}") for k in range(m)] for i in range(n)]  

        used_materials = prob.addVars(m, vtype=COPT.BINARY, nameprefix="used_material")
        prob.setObjective(cp.quicksum(used_materials[k] * X[k] * Y[k] for k in range(m)), sense=COPT.MINIMIZE)

        for i in range(n):
            prob.addConstr(cp.quicksum(material[i][k] for k in range(m)) == 1, name=f"Material_assignment_{i}")
            for k in range(m):
                prob.addConstr(material[i][k] <= used_materials[k], name=f"Link_used_material_{i}_{k}")
                prob.addConstr(x[i] + p[i] * s[i] + q[i] * (1 - s[i]) <= X[k] + X_max * (1 - material[i][k]), name=f"Bound_X_item_{i}_material_{k}")
                prob.addConstr(y[i] + q[i] * s[i] + p[i] * (1 - s[i]) <= Y[k] + Y_max * (1 - material[i][k]), name=f"Bound_Y_item_{i}_material_{k}")


        for i in range(n):
            for j in range(i + 1, n):
                for k in range(m):
                    a_ij = prob.addVar(vtype=COPT.BINARY, name=f"a_{i}_{j}_{k}")
                    b_ij = prob.addVar(vtype=COPT.BINARY, name=f"b_{i}_{j}_{k}")
                    c_ij = prob.addVar(vtype=COPT.BINARY, name=f"c_{i}_{j}_{k}")
                    d_ij = prob.addVar(vtype=COPT.BINARY, name=f"d_{i}_{j}_{k}")
                    prob.addConstr(a_ij + b_ij + c_ij + d_ij == 1, name=f"Nonoverlap_condition_{i}_{j}_material_{k}")

                    prob.addConstr(x[i] + p[i] * s[i] + q[i] * (1 - s[i]) <= x[j] + X[k] * (1 - a_ij + 1 - material[i][k] + 1 - material[j][k]), name=f"Nonoverlap_a_{i}_{j}_material_{k}")
                    prob.addConstr(x[j] + p[j] * s[j] + q[j] * (1 - s[j]) <= x[i] + X[k] * (1 - b_ij + 1 - material[i][k] + 1 - material[j][k]), name=f"Nonoverlap_b_{i}_{j}_material_{k}")
                    prob.addConstr(y[i] + q[i] * s[i] + p[i] * (1 - s[i]) <= y[j] + Y[k] * (1 - c_ij + 1 - material[i][k] + 1 - material[j][k]), name=f"Nonoverlap_c_{i}_{j}_material_{k}")
                    prob.addConstr(y[j] + q[j] * s[j] + p[j] * (1 - s[j]) <= y[i] + Y[k] * (1 - d_ij + 1 - material[i][k] + 1 - material[j][k]), name=f"Nonoverlap_d_{i}_{j}_material_{k}")


        prob.setParam(COPT.Param.Threads, 16) # Set the number of threads to use
        prob.solve()

        if prob.status == 1:  # Optimal solution  found
            solve_time = prob.getAttr(COPT.Attr.SolvingTime) 

            used_materials = [0] * m
            total_used_area = [0] * m
            cutted_stocks = []

            for i in range(n):
                w = prods[i][0] if round(s[i].x) == 1 else prods[i][1]
                h = prods[i][1] if round(s[i].x) == 1 else prods[i][0]
                assigned_material = [k for k in range(m) if round(material[i][k].x) == 1][0]
                used_materials[assigned_material] = 1  # Mark this material as used
                total_used_area[assigned_material] += w * h

            total_materials_used = sum(used_materials)
            total_waste = 0
            for k in range(m):
                if used_materials[k] == 1:
                    total_material_area = X[k] * Y[k]
                    total_waste += total_material_area - total_used_area[k]
                    cutted_stocks.append(total_used_area[k] / total_material_area)
                    print(f"""[+] Material {k}: 
        [-] Total area = {total_material_area}
        [-] Used area = {total_used_area[k]}
        [-] Waste area = {X[k] * Y[k] - total_used_area[k]}
        [-] Fill percentage: {round(total_used_area[k] / (total_used_area[k] + X[k] * Y[k] - total_used_area[k]) * 100, 2)}%""")

            info = self._get_info(cutted_stocks, stocks)
            output_file = os.path.join(self.output_folder, "output.txt")
            with open(output_file, "w") as f:
                f.write(f"Number of stocks used: {total_materials_used}\n")
                f.write(f"Filled ratio: {', '.join(f'{ratio:.2f}' for ratio in info['filled_ratio'])}\n")
                f.write(f"Trim loss: {', '.join(f'{loss:.2f}' for loss in info['trim_loss'])}\n")
                f.write(f"Solve Time: {solve_time:.2f}\n")


            self.visualize(prods, stocks, x, y, s, material, m)
            return
    
        else: 
            print(f"Solver ended with status {prob.status}")
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
    parser.add_argument("-r", action="store_true", help="generate random data")
    parser.add_argument("-m", type=int, default=5, help="number of stocks for random generation(default=5)")
    parser.add_argument("-n", type=int, default=10, help="number of products for random generation(default=10)")
    parser.add_argument("-f", type=str, help="nnput file containing stocks and products")
    parser.add_argument("-o", type=str, default="output", help="output folder to store results(default=output)")

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
