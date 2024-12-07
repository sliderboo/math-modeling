import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class OptimizedBruteForcePolicy:
    def __init__(self, stock_size):
        self.stock_size = stock_size
        self.stock = np.zeros(stock_size, dtype=int)  # 2D array to mark occupied areas
        self.product_orientations = {}
    
    def can_place(self, position, size):
        x, y = position
        width, height = size
        if x + width > self.stock_size[0] or y + height > self.stock_size[1]:
            return False
        return np.all(self.stock[x:x + width, y:y + height] == 0)
    
    def mark_stock(self, position, size):
        x, y = position
        width, height = size
        self.stock[x:x + width, y:y + height] = 1  # Mark region as occupied
    
    def rotate_product(self, size):
        # Store rotated version of product once for reuse
        width, height = size
        rotated_size = (height, width)
        return rotated_size
    
    def place_product(self, product):
        width, height = product["size"]
        
        if (width, height) not in self.product_orientations:
            self.product_orientations[(width, height)] = [
                (width, height), self.rotate_product((width, height))
            ]
        
        orientations = self.product_orientations[(width, height)]
        
        for orientation in orientations:
            w, h = orientation
            for x in range(self.stock_size[0] - w + 1):
                for y in range(self.stock_size[1] - h + 1):
                    if self.can_place((x, y), (w, h)):
                        self.mark_stock((x, y), (w, h))
                        return (x, y)
        
        return None
    def calculate_filled_percentage(self):
        # Calculate the total number of occupied cells (1's) in the stock
        occupied_cells = np.sum(self.stock)
        total_cells = self.stock_size[0] * self.stock_size[1]
        filled_percentage = (occupied_cells / total_cells) * 100
        return filled_percentage

class SkylinePolicy:
    def __init__(self, stock_size):
        self.stock_size = stock_size
        self.skyline = np.zeros(stock_size[0], dtype=int)  # Track topmost free spaces
        self.stock = np.zeros(stock_size, dtype=int)  # 2D array to mark occupied areas
    
    def can_place(self, position, size):
        x, y = position
        width, height = size
        if x + width > self.stock_size[0] or y + height > self.stock_size[1]:
            return False
        return np.all(self.stock[x:x + width, y:y + height] == 0)
    
    def mark_stock(self, position, size):
        x, y = position
        width, height = size
        self.stock[x:x + width, y:y + height] = 1  # Mark region as occupied
    
    def update_skyline(self, position, size):
        x, y = position
        width, height = size
        for i in range(width):
            self.skyline[x + i] = max(self.skyline[x + i], y + height)
    
    def place_product(self, product):
        width, height = product["size"]
        for x in range(self.stock_size[0] - width + 1):
            y = self.skyline[x]
            if y + height <= self.stock_size[1] and self.can_place((x, y), (width, height)):
                self.mark_stock((x, y), (width, height))
                self.update_skyline((x, y), (width, height))
                return (x, y)
        return None




def generate_stocks_and_products(num_stocks, max_stock_size, min_stock_size,
                                  num_products, max_product_size, min_product_size, max_quantity):
    stocks = [
        {
            "index": i,
            "size": (
                random.randint(min_stock_size[0], max_stock_size[0]),
                random.randint(min_stock_size[1], max_stock_size[1])
            )
        } for i in range(num_stocks)
    ]
    products = [
        {
            "index": i,
            "size": (
                random.randint(min_product_size[0], max_product_size[0]),
                random.randint(min_product_size[1], max_product_size[1])
            ),
            "quantity": random.randint(1, max_quantity)
        } for i in range(num_products)
    ]
    return {"stocks": stocks, "products": products}


def solve_cutting_stock_problem(data, policy_class):
    stocks = data["stocks"]
    products = data["products"]

    actions_taken = []
    policies = {}

    for stock in stocks:
        policy = policy_class(stock["size"])
        policies[stock["index"]] = policy
        for product in products:
            for _ in range(product["quantity"]):
                position = policy.place_product(product)
                if position is not None:
                    actions_taken.append({
                        "stock_idx": stock["index"],
                        "size": product["size"],
                        "position": position
                    })
                else:
                    break  # Stop trying to place this product if no space is available
    return actions_taken, policies


def visualize_stock(stock_idx, policy):
    plt.figure(figsize=(10, 10))
    plt.imshow(policy.stock.T, cmap='Greys', origin='lower', extent=[0, policy.stock_size[0], 0, policy.stock_size[1]])
    plt.title(f"Visualization of Stock {stock_idx}")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.colorbar(label="Occupancy (1=Occupied, 0=Free)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    data = generate_stocks_and_products(
        num_stocks=10,
        max_stock_size=(100, 100),
        min_stock_size=(50, 50),
        num_products=20,
        max_product_size=(30, 30),
        min_product_size=(10, 10),
        max_quantity=3
    )

    # Solve the problem using the Optimized BruteForce policy
    actions_taken_bruteforce, policies_bruteforce = solve_cutting_stock_problem(data, OptimizedBruteForcePolicy)

    print("\nActions Taken using Optimized BruteForcePolicy:")
    for action in actions_taken_bruteforce:
        print(action)

    # Visualize the stock with ID 0 using the Optimized BruteForce policy
    stock_id_to_visualize = 0
    filled_percentage = policies_bruteforce[stock_id_to_visualize].calculate_filled_percentage()
    visualize_stock(stock_id_to_visualize, policies_bruteforce[stock_id_to_visualize])
