from policy import Policy
from ortools.linear_solver import pywraplp


def Policy2352237(Policy):
    def __init__(self):
        # Student code here
        pass

    # def get_action(self, observation, info):
    #     # Student code here
    #     pass

    # Student code here
    # You can add more functions if needed
    class DelayedColumnGenerationPolicy(Policy):
        def __init__(self):
            self.solver = pywraplp.Solver.CreateSolver('GLOP')
            self.columns = []

        def get_action(self, observation, info):
            products = observation["products"]
            stocks = observation["stocks"]

            # Generate initial columns
            if not self.columns:
                self._generate_initial_columns(products, stocks)

            # Solve the restricted master problem
            solution = self._solve_master_problem(products, stocks)

            # Generate new columns if needed
            while self._generate_new_columns(solution, products, stocks):
                solution = self._solve_master_problem(products, stocks)

            # Extract the action from the solution
            action = self._extract_action(solution, products, stocks)
            return action

        def _generate_initial_columns(self, products, stocks):
            # Generate initial feasible columns
            for stock in stocks:
                for product in products:
                    if product["quantity"] > 0:
                        column = self._create_column(stock, product)
                        if column:
                            self.columns.append(column)

        def _solve_master_problem(self, products, stocks):
            # Solve the restricted master problem using the current columns
            solver = self.solver
            solver.Clear()

            # Define variables and constraints
            x = []
            for column in self.columns:
                x.append(solver.NumVar(0, solver.infinity(), 'x'))

            for product in products:
                constraint = solver.Constraint(product["quantity"], solver.infinity())
                for i, column in enumerate(self.columns):
                    if product in column["products"]:
                        constraint.SetCoefficient(x[i], 1)

            # Define the objective function
            objective = solver.Objective()
            for i, column in enumerate(self.columns):
                objective.SetCoefficient(x[i], column["cost"])
            objective.SetMinimization()

            solver.Solve()
            return x

        def _generate_new_columns(self, solution, products, stocks):
            # Generate new columns based on the dual values of the constraints
            dual_values = [constraint.dual_value() for constraint in self.solver.constraints()]
            new_columns = []

            for stock in stocks:
                for product in products:
                    if product["quantity"] > 0:
                        column = self._create_column(stock, product, dual_values)
                        if column:
                            new_columns.append(column)

            if new_columns:
                self.columns.extend(new_columns)
                return True
            return False

        def _create_column(self, stock, product, dual_values=None):
            # Create a new column for the given stock and product
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = product["size"]

            if stock_w >= prod_w and stock_h >= prod_h:
                column = {
                    "stock": stock,
                    "products": [product],
                    "cost": stock["cost"]
                }
                if dual_values:
                    column["reduced_cost"] = column["cost"] - sum(dual_values)
                return column
            return None

        def _extract_action(self, solution, products, stocks):
            # Extract the action from the solution
            for i, var in enumerate(solution):
                if var.solution_value() > 0:
                    column = self.columns[i]
                    stock_idx = stocks.index(column["stock"])
                    product = column["products"][0]
                    pos_x, pos_y = 0, 0  # Assuming placement at (0, 0) for simplicity
                    return {"stock_idx": stock_idx, "size": product["size"], "position": (pos_x, pos_y)}
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

