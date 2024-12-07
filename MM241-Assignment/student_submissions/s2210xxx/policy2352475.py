import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque
class Solution:
    class Product:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.area = width * height  # Calculate the area of the product
            self.is_balanced = self.is_balanced_product()

        def is_balanced_product(self):
            """Return True if the product is balanced (min(width, height)/max(width, height) > 0.8)"""
            return min(self.width, self.height) / max(self.width, self.height) > 0.8


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
                horizontal_right = Solution.Area(self.x + product.width, self.y, 
                                                 self.width - product.width, self.height)
                horizontal_top = Solution.Area(self.x, self.y + product.height,
                                               product.width, self.height - product.height)
                
                # Vertical cut
                vertical_right = Solution.Area(self.x + product.width, self.y,
                                               self.width - product.width, product.height)
                vertical_top = Solution.Area(self.x, self.y + product.height,
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
                new_areas.append(Solution.Area(self.x + product.width, self.y,
                                               self.width - product.width, self.height))
            elif self.height > product.height:
                # Only vertical cut possible
                new_areas.append(Solution.Area(self.x, self.y + product.height,
                                               self.width, self.height - product.height))
            
            # Sort areas by size (largest first) to promote better space utilization
            new_areas.sort(key=lambda a: a.area, reverse=True)
            return new_areas


    class Stock:
        def __init__(self, stock_id, width, height):
            self.stock_id = stock_id
            self.width = width
            self.height = height
            self.area = width * height
            self.remaining_areas = [Solution.Area(0, 0, width, height)]
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
                        new_area = Solution.Area(area1.x, area1.y, 
                                                 area1.width + area2.width, area1.height)
                        self.remaining_areas.pop(j)
                        self.remaining_areas[i] = new_area
                        continue
                        
                    # Check if areas can be merged vertically
                    if (area1.x == area2.x and area1.width == area2.width and 
                        area1.y + area1.height == area2.y):
                        new_area = Solution.Area(area1.x, area1.y,
                                                 area1.width, area1.height + area2.height)
                        self.remaining_areas.pop(j)
                        self.remaining_areas[i] = new_area
                        continue
                        
                    j += 1
                i += 1

    @staticmethod
    def place_products_across_stocks(stocks, products):
        """Improved placement strategy to maximize utilization across all stocks"""
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
            # total_area = current_stock.width * current_stock.height
            # if total_area > 0:
            #     filled_area = sum([p[2] * p[3] for p in current_stock.placed_products])
            #     filled_percentage = (filled_area / total_area) * 100
            #     print(f"Stock {current_stock.stock_id}: Filled area = {filled_percentage:.2f}%")
            # else:
            #     print(f"Stock {current_stock.stock_id}: Invalid total area (0), cannot evaluate performance.")
    @staticmethod
    def get_action(observation,info):
        input_prods = []
        input_stocks = []
        products = observation["products"]
        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                new_product = Solution.Product(prod_w, prod_h)
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
            new_stock = Solution.Stock(stock_id, stock_w, stock_h)
            stock_id += 1
            input_stocks.append(new_stock)

        # Place products across stocks
        Solution.place_products_across_stocks(input_stocks, input_prods)
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

        for placement in result:
            action = {
                "stock_idx": placement["stock_idx"],
                "size": placement["size"],
                "position": placement["position"]
            }

        return action

    