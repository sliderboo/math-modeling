import pulp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Define the Cutting Stock Problem
class CuttingStockProblem:
    def __init__(self, stocks, products, allow_rotation=True):
        """
        Initialize the Cutting Stock Problem.

        :param stocks: List of tuples representing stock sizes [(width1, height1), (width2, height2), ...]
        :param products: List of tuples representing product sizes [(width1, height1, quantity1), ...]
        :param allow_rotation: Boolean indicating if products can be rotated (swap width and height)
        """
        self.stocks = stocks
        self.products = products
        self.allow_rotation = allow_rotation
        self.model = None
        self.solution = None

    def setup_model(self):
        """
        Set up the ILP model for the Cutting Stock Problem.
        """
        self.model = pulp.LpProblem("Cutting_Stock_Problem", pulp.LpMinimize)

        # Decision Variables
        # x[p][s][r][i][j] = 1 if product p is placed on stock s with rotation r at position (i,j)
        x = {}
        for p, (pw, ph, qty) in enumerate(self.products):
            x[p] = {}
            for s, (sw, sh) in enumerate(self.stocks):
                x[p][s] = {}
                for r in [0, 1] if self.allow_rotation else [0]:
                    rotated = r == 1
                    width = ph if rotated else pw
                    height = pw if rotated else ph
                    x[p][s][r] = {}
                    for i in range(sw - width + 1):
                        for j in range(sh - height + 1):
                            var_name = f"x_{p}_{s}_{r}_{i}_{j}"
                            x[p][s][r][i, j] = pulp.LpVariable(var_name, cat='Binary')
        self.x = x

        # Objective: Minimize the number of stocks used
        y = {}
        for s in range(len(self.stocks)):
            var_name = f"y_{s}"
            y[s] = pulp.LpVariable(var_name, cat='Binary')
        self.y = y
        self.model += pulp.lpSum([y[s] for s in y]), "Minimize_Number_of_Stocks_Used"

        # Constraints

        # 1. Each product must be placed exactly as many times as its quantity
        for p, (pw, ph, qty) in enumerate(self.products):
            self.model += (
                pulp.lpSum([x[p][s][r][i, j]
                           for s in x[p]
                           for r in x[p][s]
                           for (i, j) in x[p][s][r]]) == qty,
                f"Product_{p}_Quantity"
            )

        # 2. Products placed on stocks must not overlap
        # For each stock, and each grid cell, at most one product can occupy it
        for s, (sw, sh) in enumerate(self.stocks):
            for i in range(sw):
                for j in range(sh):
                    overlapping_vars = []
                    for p, (pw, ph, qty) in enumerate(self.products):
                        for r in [0, 1] if self.allow_rotation else [0]:
                            rotated = r == 1
                            width = ph if rotated else pw
                            height = pw if rotated else ph
                            for ii in range(max(i - width + 1, 0), i + 1):
                                for jj in range(max(j - height + 1, 0), j + 1):
                                    if ii + width > sw or jj + height > sh:
                                        continue
                                    if (ii <= i < ii + width) and (jj <= j < jj + height):
                                        overlapping_vars.append(x[p][s][r].get((ii, jj), 0))
                    if overlapping_vars:
                        self.model += (
                            pulp.lpSum(overlapping_vars) <= 1,
                            f"Stock_{s}_Cell_{i}_{j}_Overlap"
                        )

        # 3. Linking Variables: If a product is placed on a stock, then that stock is used
        for p in range(len(self.products)):
            for s in range(len(self.stocks)):
                for r in [0, 1] if self.allow_rotation else [0]:
                    for (i, j) in self.x[p][s][r]:
                        self.model += (
                            self.x[p][s][r][i, j] <= self.y[s],
                            f"Link_Product_{p}_Stock_{s}_Rotation_{r}_Position_{i}_{j}"
                        )

    def solve(self):
        """
        Solve the ILP model.
        """
        if self.model is None:
            self.setup_model()
        # Solve the problem using the default solver (CBC)
        self.model.solve()
        self.solution = self.model

    def print_solution(self):
        """
        Print the solution details.
        """
        if self.solution is None:
            print("No solution found.")
            return

        if pulp.LpStatus[self.model.status] != 'Optimal':
            print("No optimal solution found.")
            return

        print(f"Number of stocks used: {pulp.value(self.model.objective)}")
        for s in range(len(self.stocks)):
            if pulp.value(self.y[s]) > 0.5:
                print(f"Stock {s} is used:")
                sw, sh = self.stocks[s]
                for p, (pw, ph, qty) in enumerate(self.products):
                    for r in [0, 1] if self.allow_rotation else [0]:
                        rotated = r == 1
                        width = ph if rotated else pw
                        height = pw if rotated else ph
                        for (i, j), var in self.x[p][s][r].items():
                            if pulp.value(var) > 0.5:
                                print(f"  Product {p} placed at ({i}, {j}) with size ({width}, {height}){' rotated' if rotated else ''}")

    def visualize_solution(self):
        """
        Visualize the solution using matplotlib.
        """
        if self.solution is None:
            print("No solution to visualize.")
            return

        if pulp.LpStatus[self.model.status] != 'Optimal':
            print("No optimal solution to visualize.")
            return

        fig, axs = plt.subplots(1, len(self.stocks), figsize=(15, 8))
        if len(self.stocks) == 1:
            axs = [axs]  # Make it iterable

        for s in range(len(self.stocks)):
            ax = axs[s]
            sw, sh = self.stocks[s]
            ax.set_xlim(0, sw)
            ax.set_ylim(0, sh)
            ax.set_title(f"Stock {s}")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            ax.set_aspect('equal')
            # Draw the stock boundary
            stock_rect = patches.Rectangle((0, 0), sw, sh, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(stock_rect)

            for p, (pw, ph, qty) in enumerate(self.products):
                for r in [0, 1] if self.allow_rotation else [0]:
                    rotated = r == 1
                    width = ph if rotated else pw
                    height = pw if rotated else ph
                    for (i, j), var in self.x[p][s][r].items():
                        if pulp.value(var) > 0.5:
                            color = plt.cm.get_cmap('tab20')(p % 20)
                            rect = patches.Rectangle((i, j), width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
                            ax.add_patch(rect)
                            ax.text(i + width/2, j + height/2, f"P{p}", ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Define Stocks (width, height)

# {'init_stocks': [(25, 23, 1), (29, 24, 1), (23, 30, 1), (24, 23, 1), (30, 29, 1)], 'init_prods': [(12, 13, 1), (7, 10, 1), (6, 9, 1), (10, 7, 1), (10, 11, 1), (14, 9, 1), (13, 9, 1), (13, 14, 1), (7, 14, 1), (10, 7, 1)]}

    stocks = [(25, 23), (29, 24), (23, 30), (24, 23), (30, 29)]

    # Define Products (width, height, quantity)
    products = [(12, 13, 1), (7, 10, 1), (6, 9, 1), (10, 7, 1), (10, 11, 1), (14, 9, 1), (13, 9, 1), (13, 14, 1), (7, 14, 1), (10, 7, 1)]


    # Initialize the problem
    problem = CuttingStockProblem(stocks=stocks, products=products, allow_rotation=True)

    # Set up the model
    problem.setup_model()

    # Solve the problem
    problem.solve()

    # Print the solution
    problem.print_solution()

    # Visualize the solution
    problem.visualize_solution()
