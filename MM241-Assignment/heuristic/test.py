import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym_cutting_stock
from abc import abstractmethod
import gymnasium as gym
import numpy as np


class Product:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.area = width * height  # Calculate the area of the product
        self.is_balanced = self.is_balanced_product()

    def is_balanced_product(self):
        """Return True if the product is balanced (min(width, height)/max(width, height) > 0.8)"""
        return min(self.width, self.height) / max(self.width, self.height) > 0.8


class Stock:
    def __init__(self, stock_id, width, height):
        self.stock_id = stock_id
        self.width = width
        self.height = height
        self.area = width * height
        self.remaining_areas = [Area(0, 0, width, height)]
        self.placed_products = []

    def place_product(self, product):
        """Try to place the product using best-fit strategy."""
        best_area = None
        best_waste = float('inf')
        best_orientation = None
        
        # Try both orientations
        orientations = [(product.width, product.height)]
        if product.width != product.height:  # Only add rotation if dimensions differ
            orientations.append((product.height, product.width))
            
        for width, height in orientations:
            for area in self.remaining_areas:
                if area.width >= width and area.height >= height:
                    waste = (area.width - width) * (area.height - height)
                    if waste < best_waste:
                        best_waste = waste
                        best_area = area
                        best_orientation = (width, height)
        
        if best_area:
            # Place the product in the best-fit area
            product.width, product.height = best_orientation
            self.placed_products.append((best_area.x, best_area.y, product.width, product.height))

            # Update remaining areas
            self.remaining_areas.remove(best_area)
            new_areas = best_area.split_area(product)
            self.remaining_areas.extend(new_areas)
            
            # Merge compatible areas
            self._merge_areas()
            return True
        
        return False

    def _merge_areas(self):
        """Merge compatible areas to reduce fragmentation."""
        i = 0
        while i < len(self.remaining_areas):
            j = i + 1
            while j < len(self.remaining_areas):
                area1 = self.remaining_areas[i]
                area2 = self.remaining_areas[j]
                
                # Check if areas can be merged horizontally
                if (area1.y == area2.y and area1.height == area2.height and 
                    area1.x + area1.width == area2.x):
                    new_area = Area(area1.x, area1.y, 
                                  area1.width + area2.width, area1.height)
                    self.remaining_areas.pop(j)
                    self.remaining_areas[i] = new_area
                    continue
                    
                # Check if areas can be merged vertically
                if (area1.x == area2.x and area1.width == area2.width and 
                    area1.y + area1.height == area2.y):
                    new_area = Area(area1.x, area1.y,
                                  area1.width, area1.height + area2.height)
                    self.remaining_areas.pop(j)
                    self.remaining_areas[i] = new_area
                    continue
                    
                j += 1
            i += 1

    def _place_product_in_area(self, product):
        """Try to place a product in the remaining areas of the stock."""
        for area in self.remaining_areas:
            if area.can_fit(product):
                # Place the product in this area
                self.placed_products.append((area.x, area.y, product.width, product.height))
                # Update remaining areas
                self.remaining_areas.remove(area)
                self.remaining_areas.extend(area.split_area(product))
                return True
        return False

class Area:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height

    def can_fit(self, product):
        return self.width >= product.width and self.height >= product.height

    def split_area(self, product):
        """Split the remaining area after placing a product using the guillotine cutting approach.
        Returns a list of new areas sorted by area size."""
        new_areas = []
        
        # Calculate the two possible cuts
        if self.width > product.width and self.height > product.height:
            # Horizontal cut
            horizontal_right = Area(self.x + product.width, self.y, 
                                 self.width - product.width, self.height)
            horizontal_top = Area(self.x, self.y + product.height,
                                product.width, self.height - product.height)
            
            # Vertical cut
            vertical_right = Area(self.x + product.width, self.y,
                                self.width - product.width, product.height)
            vertical_top = Area(self.x, self.y + product.height,
                              self.width, self.height - product.height)
            
            # Choose the cut that minimizes waste
            horizontal_waste = (horizontal_right.area if horizontal_right.area > 0 else float('inf')) + \
                             (horizontal_top.area if horizontal_top.area > 0 else float('inf'))
            vertical_waste = (vertical_right.area if vertical_right.area > 0 else float('inf')) + \
                           (vertical_top.area if vertical_top.area > 0 else float('inf'))
            
            if horizontal_waste <= vertical_waste:
                if horizontal_right.area > 0:
                    new_areas.append(horizontal_right)
                if horizontal_top.area > 0:
                    new_areas.append(horizontal_top)
            else:
                if vertical_right.area > 0:
                    new_areas.append(vertical_right)
                if vertical_top.area > 0:
                    new_areas.append(vertical_top)
        elif self.width > product.width:
            # Only horizontal cut possible
            new_areas.append(Area(self.x + product.width, self.y,
                                self.width - product.width, self.height))
        elif self.height > product.height:
            # Only vertical cut possible
            new_areas.append(Area(self.x, self.y + product.height,
                                self.width, self.height - product.height))
            
        # Sort areas by size (largest first) to promote better space utilization
        new_areas.sort(key=lambda a: a.area, reverse=True)
        return new_areas

def generate_random_stock(stock_id):
    """Generate a random stock with a random width and height."""
    width = random.randint(30, 50)
    height = random.randint(30, 50)
    return Stock(stock_id, width, height)
def generate_random_product():
    """Generate a random product with random width and height."""
    width = random.randint(5, 11)
    height = random.randint(5, 7)
    return Product(width, height)


def place_products_across_stocks(stocks, products):
    """Improved placement strategy to maximize utilization across all stocks"""
    result=[]
    # Sort products by area in descending order
    products = sorted(products, key=lambda p: p.area, reverse=True)
    unplaced_products = products.copy()
    
    while unplaced_products:
        # Find the most empty stock (lowest utilization rate)
        stock_utilizations = []
        for stock in stocks:
            filled_area = sum(p[2] * p[3] for p in stock.placed_products)
            utilization = filled_area / (stock.width * stock.height)
            stock_utilizations.append((stock, utilization))
        
        # Sort stocks by utilization (least utilized first)
        stock_utilizations.sort(key=lambda x: x[1])
        
        # Try to place products in the most empty stock
        current_stock = stock_utilizations[0][0]
        products_to_remove = []
        
        # Try to fill current stock with remaining products
        for product in unplaced_products:
            if current_stock.place_product(product):
                products_to_remove.append(product)
                
        # Remove placed products from unplaced list
        for product in products_to_remove:
            unplaced_products.remove(product)
            
        # If we couldn't place any products in this iteration, break to avoid infinite loop
        if not products_to_remove:
            print(f"Warning: Could not place {len(unplaced_products)} remaining products")
            break
            
        # Evaluate performance for the current stock
        evaluate_performance(current_stock)
        #print(current_stock.placed_products)
        #visualize_stock(current_stock)


def generate_random_color():
    """Generate a random color in RGB format."""
    return (random.random(), random.random(), random.random())

def visualize_stock(stock):
    # Create a new figure
    fig, ax = plt.subplots()

    # Add the main stock area as a rectangle (consider stock.width and stock.height)
    ax.add_patch(patches.Rectangle((0, 0), stock.width, stock.height, linewidth=1, edgecolor='black', facecolor='lightgray'))

    # Plot the placed products in the stock with random colors
    for product in stock.placed_products:
        # Create a random color for each product
        color = generate_random_color()
        x, y, width, height = product
        ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color))

    # Set limits and labels
    ax.set_xlim(0, stock.width)
    ax.set_ylim(0, stock.height)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    # Set aspect ratio to ensure square units
    plt.gca().set_aspect('equal', adjustable='box')

    # Title
    ax.set_title(f"Stock {stock.stock_id}: Utilization = {sum([p[2]*p[3] for p in stock.placed_products]) / (stock.width * stock.height) * 100:.2f}%")

    # Show the plot
    plt.show()
    plt.close()


def evaluate_performance(stock):
    total_area = stock.width * stock.height
    filled_area = sum([product[2] * product[3] for product in stock.placed_products])
    filled_percentage = (filled_area / total_area) * 100
    print(f"Stock {stock.stock_id}: Filled area = {filled_percentage:.2f}%")

# Example Usage
if __name__ == "__main__":
    # Initialize the environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    observation, info = env.reset(seed=42)

    # Prepare input data
    input_prods = []
    input_stocks = []

    products = observation["products"]
    for prod in products:
        if prod["quantity"] > 0:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            new_product = Product(prod_w, prod_h)
            for _ in range(prod["quantity"]):
                input_prods.append(new_product)

    stocks = observation["stocks"]
    def _get_stock_size_(stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    stock_id = 0
    for stock in stocks:
        stock_w, stock_h = _get_stock_size_(stock)
        new_stock = Stock(stock_id, stock_w, stock_h)
        stock_id += 1
        input_stocks.append(new_stock)

    # Place products across stocks
    place_products_across_stocks(input_stocks, input_prods)

    result = []
    for stock_idx, stock in enumerate(input_stocks):
        # Iterate over the placed products in the current stock
        for product in stock.placed_products:
            pos_x, pos_y, prod_width, prod_height = product
            result.append({
                "stock_idx": stock_idx,  # Index of the current stock
                "size": (prod_width, prod_height), 
                "position": (pos_x, pos_y)
            })

    # Now interact with the environment for each product placement
    for placement in result:
        # Define action format based on how the environment expects the input
        action = {
            "stock_idx": placement["stock_idx"],
            "size": placement["size"],
            "position": placement["position"]
        }
        print(action)
        # Perform the step with the action
        observation, reward, done, truncated, info = env.step(action)
        print(f"Step result: {observation}, Reward: {reward}")

    