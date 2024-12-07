from policy import Policy
import numpy as np


class Policy2210xxx(Policy):
    def __init__(self):
        # Initialization (can add any parameters if needed)
        self.best_solution = None  # To store the best solution found so far
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]
        
        best_score = float('inf')  # Initialize the best score as a very large number
        best_action = None  # The best action (stock index, position, product size)
        # Branch and Bound approach
        i = 0
        for stock_idx, stock in enumerate(list_stocks):
            print(i)
            i+=1
            stock_w, stock_h = self._get_stock_size_(stock)
            

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
        print(best_action)
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
