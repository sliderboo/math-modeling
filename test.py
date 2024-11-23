from ortools.linear_solver import pywraplp
import numpy as np

def solve_2d_cutting_stock():
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('CBC')  # Change solver to 'CBC', a Mixed Integer Programming solver

    # print(help(solver))

    if not solver:
        return None

    # Data for the problem
    # List of available large sheets (width, height)
    # Data for the problem
    # List of available large sheets (width, height)
    stocks = [(10, 12), (21, 10), (7, 8), (6, 6), (5, 10)]

    # Small items to cut (width, height, demand)
    prods = [
        (3, 5, 14),  # (width, height, demand)
        (2, 3, 16),
        (5, 2, 13),
    ]

    # Variables: x[i][j] represents the number of times we cut item i from stock j
    x = []
    for i in range(len(prods)):
        x.append([solver.IntVar(0, solver.infinity(), f'x_{i}_{j}') for j in range(len(stocks))])

    # Variables: y[j] represents whether stock j is used (binary variable)
    y = []
    for j in range(len(stocks)):
        y.append(solver.BoolVar(f'y_{j}'))

    # Constraints
    # Ensure that we meet the demand for each item
    for i in range(len(prods)):
        solver.Add(sum(x[i][j] for j in range(len(stocks))) >= prods[i][2])

    # Ensure that the total area used in each stock does not exceed the available area of that stock
    for j in range(len(stocks)):
        stock_width, stock_height = stocks[j]
        total_used_area = sum(x[i][j] * prods[i][0] * prods[i][1] for i in range(len(prods)))
        solver.Add(total_used_area <= y[j] * stock_width * stock_height)

    # Objective: Minimize the total number of large sheets used
    objective = solver.Objective()
    for j in range(len(stocks)):
        objective.SetCoefficient(y[j], 1)
    objective.SetMinimization()

    # Solve the problem
    status = solver.Solve()

    # Check if a solution was found
    if status == pywraplp.Solver.OPTIMAL:

    ################# DEBUG BLOCK #################
        print('Solution found:')
        total_sheets_used = 0
        for j in range(len(stocks)):
            if y[j].solution_value() == 1:
                total_sheets_used += 1
                sheet_details = []
                for i in range(len(prods)):
                    if x[i][j].solution_value() > 0:
                        sheet_details.append(f'{int(x[i][j].solution_value())}({prods[i][0]}x{prods[i][1]})')
                print(f'Sheet {j}: ' + ', '.join(sheet_details))
        print(f'Number of large sheets used: {total_sheets_used}')
        
        for i in range(len(prods)):
            total_cuts = sum(x[i][j].solution_value() for j in range(len(stocks)))
            print(f'Cut {total_cuts} pieces of item {i} (size {prods[i][0]}x{prods[i][1]})')
        
        total_used_area_value = sum(x[i][j].solution_value() * prods[i][0] * prods[i][1] for i in range(len(prods)) for j in range(len(stocks)))
        print(f'Total used area: {total_used_area_value}')
        total_waste = sum(y[j].solution_value() * (stocks[j][0] * stocks[j][1]) for j in range(len(stocks))) - total_used_area_value
        print(f'Total waste: {total_waste}')
    ################# DEBUG BLOCK #################

    else:
        print('No solution found')

if __name__ == '__main__':
    solve_2d_cutting_stock()
