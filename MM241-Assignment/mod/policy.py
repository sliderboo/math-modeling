import random
from abc import abstractmethod
from time import sleep
from ortools.linear_solver import pywraplp
import numpy as np

class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

class RandomPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Random choice a stock idx
                pos_x, pos_y = None, None
                for _ in range(100):
                    # random choice a stock
                    stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                    stock = observation["stocks"][stock_idx]

                    # Random choice a position
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)

                    if not self._can_place_(stock, (pos_x, pos_y), prod_size):
                        continue

                    break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class GreedyPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class GreedyPolicy2(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Iterate over all products to find the one with the highest demand
        sorted_prods = sorted(list_prods, key=lambda p: p["quantity"], reverse=True)
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Iterate over all stocks to find the first available position
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Iterate through possible positions row by row
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                stock_idx = idx
                                pos_x, pos_y = x, y
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class BestFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        min_waste = float('inf')
        best_action = None

        # Iterate over all products to find the one with the highest demand
        sorted_prods = sorted(list_prods, key=lambda p: p["quantity"], reverse=True)
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Iterate over all stocks to find the position that minimizes waste
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Iterate through possible positions row by row
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                remaining_width = stock_w - (x + prod_w)
                                remaining_height = stock_h - (y + prod_h)
                                waste = remaining_width * remaining_height

                                if waste < min_waste:
                                    min_waste = waste
                                    best_action = {"stock_idx": idx, "size": prod_size, "position": (x, y)}

        if best_action:
            return best_action

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class BestFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        min_waste = float('inf')
        best_action = None

        # Iterate over all products to find the one with the highest demand
        sorted_prods = sorted(list_prods, key=lambda p: p["quantity"], reverse=True)
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Iterate over all stocks to find the position that minimizes waste
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Iterate through possible positions row by row
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                remaining_width = stock_w - (x + prod_w)
                                remaining_height = stock_h - (y + prod_h)
                                waste = remaining_width * remaining_height

                                if waste < min_waste:
                                    min_waste = waste
                                    best_action = {"stock_idx": idx, "size": prod_size, "position": (x, y)}

        if best_action:
            return best_action

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class FirstFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Iterate over all products to find the one with the highest demand
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Iterate over all stocks to find the first available position
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Iterate through possible positions row by row
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                stock_idx = idx
                                pos_x, pos_y = x, y
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class WorstFitPolicy(Policy):
    def __init__(self):
        self.list_prods = []
        self.stocks = []
        self.original_stocks = []

    def get_action(self, observation, info):
        # Store products and stocks to self attributes
        self.list_prods = observation["products"]
        self.stocks = observation["stocks"]
        self.original_stocks = list(enumerate(self.stocks))

        # Sort products and stocks by height * width in decreasing order
        self.list_prods = sorted(self.list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        self.stocks = sorted(self.original_stocks, key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1], reverse=True)


        # Iterate over all stocks from largest to smallest
        for sorted_idx, (original_idx, stock) in enumerate(self.stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # Iterate over all products sorted by size to fill the current stock
            for prod in self.list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Skip the current stock if it's not large enough for the product
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Iterate through possible positions row by row
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": original_idx, "size": prod_size, "position": (x, y)}

        # If no valid placement found, return default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class DCGPolicy(Policy):
    def __init__(self):
        """
        Initializes the Delayed Column Generation Policy for the 2D Cutting Stock Problem.
        """
        # Initialize data structures to store cutting patterns
        self.patterns = []  # List of patterns, each pattern is a list of (prod_idx, count)
        self.solver: pywraplp.Solver = pywraplp.Solver.CreateSolver('SCIP')  # Using SCIP solver
        if not self.solver:
            raise Exception("Solver not found.")

        # Placeholder for solver variables and constraints
        self.variables = []
        self.constraints = []
        self.obj = None

    def get_action(self, observation, info):
        """
        Determines the next action using the Delayed Column Generation algorithm.

        Parameters:
            observation (dict): Contains 'products' and 'stocks'.
            info (dict): Additional information (not used here).

        Returns:
            dict: Contains 'stock_idx', 'size', and 'position'.
        """
        # Extract products and stocks from observation
        products = observation["products"]
        stocks = observation["stocks"]

        # Step 1: Initialize or update the master problem with existing patterns
        self._setup_master_problem(products, stocks)


        # Step 2: Solve the master problem to get dual prices
        master_status, duals = self._solve_master_problem()

        print (str(products))
        print (str(duals))
        exit()

        # if master_status != pywraplp.Solver.OPTIMAL:
        #     print("Master problem did not find an optimal solution.")
        #     return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Step 3: Solve the subproblem to generate a new pattern
        new_pattern, reduced_cost = self._solve_subproblem(products, duals)

        # Step 4: Check if the new pattern has negative reduced cost
        if reduced_cost < -1e-5:
            # Add the new pattern to the master problem
            self._add_pattern(new_pattern, products)
            # Re-solve the master problem with the new pattern
            master_status, duals = self._solve_master_problem()
            if master_status != pywraplp.Solver.OPTIMAL:
                print("Master problem did not find an optimal solution after adding new pattern.")
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        else:
            print("No improving patterns found. Current solution is optimal.")

        # Step 5: Extract the selected patterns and determine the action
        selected_pattern = self._extract_selected_pattern()
        if selected_pattern is None:
            print("No pattern selected.")
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Determine the placement based on the selected pattern
        action = self._determine_action_from_pattern(selected_pattern, stocks)
        return action

    def _setup_master_problem(self, products, stocks):
        """
        Sets up or updates the master problem with existing patterns.

        Parameters:
            products (list): List of products with 'size' and 'quantity'.
            stocks (list): List of stocks (2D arrays).
        """
        # If this is the first time, define variables and constraints
        if not self.patterns:
            # Initially, no patterns are defined. We'll generate them in the subproblem.
            pass
        else:
            # Update constraints based on product quantities
            for idx, prod in enumerate(products):
                constraint = self.constraints[idx]
                constraint.SetBounds(prod["quantity"], prod["quantity"])
                # Update coefficients for variables based on patterns
                for var, pattern in zip(self.variables, self.patterns):
                    var.SetCoefficient(var, pattern.count(idx))
        # Define constraints for each product
        if not self.constraints:
            for prod in products:
                constraint = self.solver.Constraint(prod["quantity"], prod["quantity"])
                self.constraints.append(constraint)
        # Define the objective to minimize the number of stocks used
        if not self.obj:
            self.obj = self.solver.Objective()
            self.obj.SetMinimization()
            for var in self.variables:
                self.obj.SetCoefficient(var, 1)

    def _solve_master_problem(self):
        """
        Solves the master problem and retrieves dual prices.

        Returns:
            tuple: (status, duals)
        """
        status = self.solver.Solve()

        # Retrieve dual prices
        duals = [constraint.dual_value() for constraint in self.constraints]
        return status, duals

    def _solve_subproblem(self, products, duals):
        """
        Solves the subproblem (pricing problem) to find a new pattern.

        Parameters:
            products (list): List of products.
            duals (list): Dual prices from the master problem.

        Returns:
            tuple: (new_pattern, reduced_cost)
        """
        # The subproblem aims to find a pattern that minimizes (1 - sum(duals * counts))
        # where counts are the number of each product in the pattern.

        # Initialize subproblem solver
        sub_solver = pywraplp.Solver.CreateSolver('SCIP')
        if not sub_solver:
            raise Exception("Subproblem solver not found.")

        num_products = len(products)
        x = [sub_solver.IntVar(0, sub_solver.infinity(), f'x_{i}') for i in range(num_products)]

        # Objective: minimize (1 - sum(dual_i * x_i))
        objective = sub_solver.Objective()
        for i in range(num_products):
            objective.SetCoefficient(x[i], -duals[i])
        objective.SetOffset(1)
        objective.SetMinimization()

        # Constraints: The pattern must fit within at least one stock
        # For simplicity, we consider a single stock type here. Extend as needed.
        # Assuming all stocks are identical in size; otherwise, handle multiple types.

        # Here, implement constraints to ensure that the total size of products in the pattern
        # does not exceed the stock dimensions. This requires a more detailed model,
        # possibly involving 2D bin packing constraints.

        # Placeholder: No constraints other than non-negativity
        # Implementing full 2D packing constraints is complex and beyond this example.

        # For demonstration, we assume a simplistic 1D bin packing
        # Replace with proper 2D constraints as needed.

        # Example constraint: sum(width_i * x_i) <= stock_width
        # Here, we assume the first stock's width and height
        if stocks := products[0].get("stock"):
            stock_width, stock_height = self._get_stock_size_(stocks)
        else:
            # Default stock size if not provided
            stock_width, stock_height = 100, 100

        total_width = sum(products[i]["size"][0] * x[i] for i in range(num_products))
        total_height = sum(products[i]["size"][1] * x[i] for i in range(num_products))

        sub_solver.Add(total_width <= stock_width)
        sub_solver.Add(total_height <= stock_height)

        # Solve the subproblem
        sub_status = sub_solver.Solve()
        if sub_status != pywraplp.Solver.OPTIMAL:
            # No improving pattern found
            return None, 0

        # Extract the new pattern
        new_pattern = []
        for i in range(num_products):
            count = int(x[i].solution_value())
            if count > 0:
                new_pattern.append((i, count))

        # Calculate reduced cost
        reduced_cost = sub_solver.Objective().Value()

        return new_pattern, reduced_cost

    def _add_pattern(self, pattern, products):
        """
        Adds a new cutting pattern to the master problem.

        Parameters:
            pattern (list): List of tuples (prod_idx, count).
            products (list): List of products.
        """
        self.patterns.append(pattern)
        # Create a new variable for the pattern
        var = self.solver.IntVar(0, self.solver.infinity(), f'pattern_{len(self.patterns)-1}')
        self.variables.append(var)
        # Update the objective
        self.obj.SetCoefficient(var, 1)
        # Update constraints based on the pattern
        for prod_idx, count in pattern:
            self.constraints[prod_idx].SetCoefficient(var, count)

    def _extract_selected_pattern(self):
        """
        Extracts the pattern selected by the master problem.

        Returns:
            list: Selected pattern as a list of (prod_idx, count).
        """
        # Find the pattern with a non-zero variable
        for var, pattern in zip(self.variables, self.patterns):
            if var.solution_value() > 0.5:  # Assuming integer solution
                return pattern
        return None

    def _determine_action_from_pattern(self, observation, pattern, stocks):
        """
        Determines the placement action based on the selected pattern.

        Parameters:
            pattern (list): Selected pattern as a list of (prod_idx, count).
            stocks (list): List of stocks (2D arrays).

        Returns:
            dict: Action containing 'stock_idx', 'size', and 'position'.
        """
        # For simplicity, select the first product in the pattern
        if not pattern:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        prod_idx, count = pattern[0]
        product = observation["products"][prod_idx]
        prod_size = product["size"]

        # Attempt to place the product in the first available stock
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            if stock_w < prod_w or stock_h < prod_h:
                continue

            # Iterate through possible positions
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        # Update the stock grid to mark the placement
                        stock[x:x+prod_w, y:y+prod_h] = prod_idx
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

        # If no placement is found, return a failure action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
class LPPolicy(Policy):
    def __init__(self):
        """
        Initializes the Integer Programming Policy for the 2D Cutting Stock Problem.
        """
        super().__init__()
        self.actions = []  # List to store placement actions

    def get_action(self, observation, info):
        """
        Determines the next action using Integer Programming.

        Parameters:
            observation (dict): Contains 'products' and 'stocks'.
            info (dict): Additional information (not used here).

        Returns:
            dict: Contains 'stock_idx', 'size', and 'position'.
        """
        if not self.actions:
            # Solve the IP model to determine all placements
            self.actions = self._solve_integer_programming(observation)
        
        if self.actions:
            return self.actions.pop(0)
        else:
            # No more actions to perform
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _solve_integer_programming(self, observation):
        """
        Formulates and solves the Integer Programming model to determine product placements.

        Parameters:
            observation (dict): Contains 'products' and 'stocks'.

        Returns:
            list: List of placement actions.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        num_products = len(products)
        num_stocks = len(stocks)

        # Initialize the solver
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise Exception("CBC solver not found.")

        # Variables
        # y[s] = 1 if stock s is used
        y = [solver.BoolVar(f'y_{s}') for s in range(num_stocks)]

        print (str(y))

        # x[p][s][i][j] = 1 if product p is placed in stock s at position (i, j)
        x = {}
        for p, product in enumerate(products):
            x[p] = {}
            width_p, height_p = product["size"]
            quantity_p = product["quantity"]
            print ("Width: " + str(width_p) + " Height: " + str(height_p) + " Quantity: " + str(quantity_p))
            for s, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                for i in range(stock_w - width_p + 1):
                    for j in range(stock_h - height_p + 1):
                        var = solver.BoolVar(f'x_{p}_{s}_{i}_{j}')
                        print (str(var))
                        x[p][s, i, j] = var

        exit()
        # Constraints

        # 1. Each product must be placed exactly once
        for p in range(num_products):
            constraint = solver.Constraint(1, 1, f'Product_{p}_placement')
            for s in range(num_stocks):
                stock = stocks[s]
                stock_w, stock_h = self._get_stock_size_(stock)
                width_p, height_p = products[p]["size"]
                for i in range(stock_w - width_p + 1):
                    for j in range(stock_h - height_p + 1):
                        if (s, i, j) in x[p]:
                            constraint.SetCoefficient(x[p][s, i, j], 1)

        # 2. No overlapping: Each cell in a stock can be occupied by at most one product
        for s, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for i in range(stock_w):
                for j in range(stock_h):
                    constraint = solver.Constraint(0, 1, f'No_overlap_stock_{s}_cell_{i}_{j}')
                    for p, product in enumerate(products):
                        width_p, height_p = product["size"]
                        # Check if cell (i, j) is covered by a placement of product p
                        for dx in range(width_p):
                            for dy in range(height_p):
                                xi = i - dx
                                yj = j - dy
                                if 0 <= xi < (stock_w - width_p + 1) and 0 <= yj < (stock_h - height_p + 1):
                                    if (s, xi, yj) in x[p]:
                                        constraint.SetCoefficient(x[p][s, xi, yj], 1)

        # 3. Linking stock usage with product placements
        for s in range(num_stocks):
            for p in range(num_products):
                product = products[p]
                width_p, height_p = product["size"]
                stock_w, stock_h = self._get_stock_size_(stocks[s])
                for i in range(stock_w - width_p + 1):
                    for j in range(stock_h - height_p + 1):
                        if (s, i, j) in x[p]:
                            # If product p is placed in stock s at (i, j), then y[s] must be 1
                            solver.Add(y[s] >= x[p][s, i, j])

        # Objective: Minimize the number of stocks used
        objective = solver.Objective()
        for s in range(num_stocks):
            objective.SetCoefficient(y[s], 1)
        objective.SetMinimization()

        # Solve the model
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            print("No optimal solution found.")
            return []

        # Extract placements from the solution
        placements = []
        for p in range(num_products):
            placed = False
            for s in range(num_stocks):
                stock = stocks[s]
                stock_w, stock_h = self._get_stock_size_(stock)
                width_p, height_p = products[p]["size"]
                for i in range(stock_w - width_p + 1):
                    for j in range(stock_h - height_p + 1):
                        if (s, i, j) in x[p]:
                            if x[p][s, i, j].solution_value() > 0.5:
                                placements.append({
                                    "stock_idx": s,
                                    "size": products[p]["size"],
                                    "position": (i, j)
                                })
                                placed = True
                                break
                    if placed:
                        break
                if placed:
                    break
        return placements
    
class CBCPolicy(Policy):
    def __init__(self):
        self.stocks = []
        self.prods = []
        self.placements = []
        self.placements_size = 0
        self.is_solved = False  
        self.current_positions = 0


    def solve_2d_cutting_stock(self):
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('CBC')  # Change solver to 'CBC', a Mixed Integer Programming solver
        if not solver:
            return None
        
        # Variables: x[i][j] represents the number of times we cut item i from stock j
        x = []
        for i in range(len(self.prods)):
            x.append([solver.IntVar(0, solver.infinity(), f'x_{i}_{j}') for j in range(len(self.stocks))])

        # Variables: y[j] represents whether stock j is used (binary variable)
        y = []
        for j in range(len(self.stocks)):
            y.append(solver.BoolVar(f'y_{j}'))

        # Constraints
        # Ensure that we meet the demand for each item
        for i in range(len(self.prods)):
            solver.Add(sum(x[i][j] for j in range(len(self.stocks))) >= self.prods[i][2])

        # Ensure that the total area used in each stock does not exceed the available area of that stock
        for j in range(len(self.stocks)):
            stock_width, stock_height = self.stocks[j]
            total_used_area = sum(x[i][j] * self.prods[i][0] * self.prods[i][1] for i in range(len(self.prods)))
            solver.Add(total_used_area <= y[j] * stock_width * stock_height)

        # Objective: Minimize the total number of large sheets used
        objective = solver.Objective()
        for j in range(len(self.stocks)):
            objective.SetCoefficient(y[j], 1)
        objective.SetMinimization()

        # Solve the problem
        status = solver.Solve()

        placement_details = []

        # Check if a solution was found
        if status == pywraplp.Solver.OPTIMAL:


            ################# DEBUG BLOCK #################
            print('Solution found:')
            total_sheets_used = 0
            for j in range(len(self.stocks)):
                if y[j].solution_value() == 1:
                    total_sheets_used += 1
                    sheet_details = []
                    for i in range(len(self.prods)):
                        if x[i][j].solution_value() > 0:
                            sheet_details.append(f'{int(x[i][j].solution_value())}({self.prods[i][0]}x{self.prods[i][1]})')
                    print(f'Sheet {j}: ' + ', '.join(sheet_details))
            print(f'Number of large sheets used: {total_sheets_used}')
            
            for i in range(len(self.prods)):
                total_cuts = sum(x[i][j].solution_value() for j in range(len(self.stocks)))
                print(f'Cut {total_cuts} pieces of item {i} (size {self.prods[i][0]}x{self.prods[i][1]})')
            
            total_used_area_value = sum(x[i][j].solution_value() * self.prods[i][0] * self.prods[i][1] for i in range(len(self.prods)) for j in range(len(self.stocks)))
            print(f'Total used area: {total_used_area_value}')
            total_waste = sum(y[j].solution_value() * (self.stocks[j][0] * self.stocks[j][1]) for j in range(len(self.stocks))) - total_used_area_value
            print(f'Total waste: {total_waste}')
            ################# DEBUG BLOCK #################



            current_positions = [(0, 0) for _ in range(len(self.stocks))]  # Track current x, y positions for each stock

            for j in range(len(self.stocks)):
                if y[j].solution_value() == 1:
                    stock_width, stock_height = self.stocks[j]
                    current_x, current_y = current_positions[j]

                    for i in range(len(self.prods)):
                        for _ in range(int(x[i][j].solution_value())):
                            width, height = self.prods[i][0], self.prods[i][1]

                            # Check if we need to move to a new row
                            if current_x + width > stock_width:
                                current_x = 0
                                current_y += height

                            # If the item doesn't fit in the current stock, move to next stock
                            if current_y + height > stock_height:
                                break

                            # Record the placement details
                            placement_details.append({
                                "stock_idx": j,
                                "size": np.array([width, height]),
                                "position": (current_x, current_y)
                            })

                            # Update the current position
                            current_x += width

                    # Update the current positions
                    current_positions[j] = (current_x, current_y)

        else:
            print('No solution found')

        return placement_details

    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        self.stocks = [self._get_stock_size_(stock) for stock in list_stocks]
        self.prods = [(prod["size"][0], prod["size"][1], prod["quantity"]) for prod in list_prods]

        if not self.is_solved:
            self.placements = self.solve_2d_cutting_stock()
            self.is_solved = True
            self.placements_size = len(self.placements)
        
        if self.placements and self.current_positions < self.placements_size:
            tmp = self.current_positions
            self.current_positions += 1
            return self.placements[tmp]
        else:
            return None

class NextFitDecreasingPolicy(Policy):
    def __init__(self):
        # Keep track of the last used stock index
        self.last_stock_idx = 0

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sort products by size (area) in decreasing order
        sorted_prods = sorted(list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Start checking from the last used stock index
                for offset in range(len(stocks)):
                    idx = (self.last_stock_idx + offset) % len(stocks)
                    stock = stocks[idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Iterate through possible positions row by row
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                stock_idx = idx
                                pos_x, pos_y = x, y
                                # Place the product and update the last used stock index
                                self.last_stock_idx = idx
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class ExactLinearPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return None

        products = observation["products"]
        stocks = observation["stocks"]
        n_products = len(products)
        n_stocks = len(stocks)

        # Variables: x[i][j] represents whether product i is placed in stock j
        x = []
        for i in range(n_products):
            x.append([solver.BoolVar(f'x_{i}_{j}') for j in range(n_stocks)])

        # Variables: y[j] represents whether stock j is used (binary variable)
        y = [solver.BoolVar(f'y_{j}') for j in range(n_stocks)]

        # Objective: Minimize the number of stocks used
        objective = solver.Objective()
        for j in range(n_stocks):
            objective.SetCoefficient(y[j], 1)
        objective.SetMinimization()

        # Constraints
        # Each product must be placed in exactly one stock
        for i in range(n_products):
            solver.Add(sum(x[i][j] for j in range(n_stocks)) == 1)

        # Ensure that the products fit within the stocks without overlapping
        for j in range(n_stocks):
            stock_w, stock_h = stocks[j].shape
            for i in range(n_products):
                prod_w, prod_h = products[i]["size"]
                if prod_w > stock_w or prod_h > stock_h:
                    # If the product doesn't fit in the stock, we can't place it there
                    solver.Add(x[i][j] == 0)

        # Solve the problem
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            solution = []
            for j in range(n_stocks):
                if y[j].solution_value() == 1:
                    stock_details = []
                    for i in range(n_products):
                        if x[i][j].solution_value() == 1:
                            stock_details.append((i, products[i]["size"]))
                    solution.append({"stock_idx": j, "products": stock_details})

            print(str(solution))
            exit(0)
            return solution
        else:
            print("No solution found.")
            return None

class BieuPolicy(Policy):
    def __init__(self):
        # Initialization (can add any parameters if needed)
        self.best_solution = None  # To store the best solution found so far
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]
        
        best_score = float('inf')  # Initialize the best score as a very large number
        best_action = None  # The best action (stock index, position, product size)

        print (str(list_prods))
        
        # Branch and Bound approach
        for stock_idx, stock in enumerate(list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            print("debug")
            for prod in list_prods:
                if prod["quantity"] > 0:  # We can only consider products that need cutting
                    prod_w, prod_h = prod["size"]
                    
                    # Try both orientations (no rotation and rotated)
                    for prod_size in [(prod_w, prod_h), (prod_h, prod_w)]:
                        prod_w, prod_h = prod_size

                        
                        # Check if the stock can accommodate the product
                        if stock_w >= prod_w and stock_h >= prod_h:
                            # Try to place the product in all valid positions
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        # If it can fit, evaluate this solution
                                        new_score = self.evaluate_cut(stock, x, y, prod_size, list_prods)
                                        if new_score < best_score:
                                            best_score = new_score
                                            best_action = {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}
        
        # Return the best action found
        return best_action
    
    def evaluate_cut(self, stock, x, y, prod_size, list_prods):
        """
        Evaluate the current cut by calculating the total remaining waste in the stock.
        This is where the branch and bound 'bounding' happens.
        """
        # Create a copy of the stock to simulate the cut
        stock_copy = stock.copy()
        prod_w, prod_h = prod_size
        
        # Mark the area as used (cut the product)
        stock_copy[x:x+prod_w, y:y+prod_h] = -2
        
        # Calculate the trim loss (unused area in the stock)
        remaining_area = np.sum(stock_copy == -1)
        
        # We can use this area as a heuristic to bound the search space
        # A lower trim loss means a better solution
        return remaining_area
    






