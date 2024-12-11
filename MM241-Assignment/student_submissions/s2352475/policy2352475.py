import numpy as np
import time
from policy import Policy
class Policy2352475(Policy):
    def __init__(self):
        self.isSolved= False
        self.result=[]
        self.performance=[]
        self.it=0
        pass

    def get_action(self, observation, info):
        if self.it >= len(self.result):
            self.isSolved = False
            self.it = 0
            self.result = []

        if not self.isSolved:
            input_prods = []
            input_stocks = []
            products = observation["products"]
            for prod in products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    new_product = Policy2352475.Product(prod_w, prod_h)
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
                new_stock = Policy2352475.Stock(stock_id, stock_w, stock_h)
                stock_id += 1
                input_stocks.append(new_stock)
            start_time = time.time()
            Policy2352475.place_products_across_stocks(input_stocks, input_prods)
            end_time = time.time()
            time_to_solve = end_time - start_time
            # Call evaluate_performance and capture the returned metrics
            performance_metrics = Policy2352475.evaluate_performance(input_stocks)
            if performance_metrics:
                performance_metrics['time_to_solve'] = time_to_solve
                self.performance.append(performance_metrics)


            self.isSolved = True 
            for stock_idx, stock in enumerate(input_stocks):
                # Iterate over the placed products in the current stock
                for product in stock.placed_products:
                    pos_x, pos_y, prod_width, prod_height = product
                    self.result.append({
                        "stock_idx": stock_idx,  # Index of the current stock
                        "size": (prod_width, prod_height), 
                        "position": (pos_x, pos_y)
                    })
    
        action = {
            "stock_idx": self.result[self.it]["stock_idx"],
            "size": self.result[self.it]["size"],
            "position": self.result[self.it]["position"]
        }
        self.it += 1
        return action
    

    class Product:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.area = width * height  # Calculate the area of the product
            
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
                horizontal_right = Policy2352475.Area(self.x + product.width, self.y, 
                                                 self.width - product.width, self.height)
                horizontal_top = Policy2352475.Area(self.x, self.y + product.height,
                                               product.width, self.height - product.height)
                
                # Vertical cut
                vertical_right = Policy2352475.Area(self.x + product.width, self.y,
                                               self.width - product.width, product.height)
                vertical_top = Policy2352475.Area(self.x, self.y + product.height,
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
                new_areas.append(Policy2352475.Area(self.x + product.width, self.y,
                                               self.width - product.width, self.height))
            elif self.height > product.height:
                # Only vertical cut possible
                new_areas.append(Policy2352475.Area(self.x, self.y + product.height,
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
            self.remaining_areas = [Policy2352475.Area(0, 0, width, height)]
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
                        new_area = Policy2352475.Area(area1.x, area1.y, 
                                                 area1.width + area2.width, area1.height)
                        self.remaining_areas.pop(j)
                        self.remaining_areas[i] = new_area
                        continue
                        
                    # Check if areas can be merged vertically
                    if (area1.x == area2.x and area1.width == area2.width and 
                        area1.y + area1.height == area2.y):
                        new_area = Policy2352475.Area(area1.x, area1.y,
                                                 area1.width, area1.height + area2.height)
                        self.remaining_areas.pop(j)
                        self.remaining_areas[i] = new_area
                        continue
                        
                    j += 1
                i += 1
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
    def evaluate_performance(stocks):
        """
        Evaluate and print aggregate performance metrics for all stocks.

        Metrics:
        - Maximum filled area (%)
        - Minimum filled area (%)
        - Trim loss: Unused area across all filled stocks (%)
        - Unused stocks: Number of stocks with 0% filled area
        """
        utilized_stocks = [stock for stock in stocks if sum(product[2] * product[3] for product in stock.placed_products) > 0]
        
        total_area = sum(stock.width * stock.height for stock in utilized_stocks)  # Total area of utilized stocks
        
        # Calculate the filled areas for each utilized stock
        filled_areas = [
            sum(product[2] * product[3] for product in stock.placed_products)
            for stock in utilized_stocks
        ]
        
        # Calculate trim loss as the sum of unused areas in utilized stocks
        trim_loss = sum((stock.width * stock.height - filled_area) for stock, filled_area in zip(utilized_stocks, filled_areas))
        
        # Calculate the filled percentages for utilized stocks
        filled_percentages = [
            (filled_area / (stock.width * stock.height)) * 100
            for filled_area, stock in zip(filled_areas, utilized_stocks)
        ]
        
        # Calculate the number of unused stocks (those with zero filled area)
        unused_stock_count = len([stock for stock in stocks if sum(product[2] * product[3] for product in stock.placed_products) == 0])
        
        # Calculate max, min, and average filled areas
        max_filled_area = np.amax(filled_percentages) if filled_percentages else 0
        min_filled_area = np.amin(filled_percentages) if filled_percentages else 0
        avg_filled_area = np.mean(filled_percentages) if filled_percentages else 0
    
        
        # Calculate trim loss as a percentage
        trim_loss_percentage = (trim_loss / total_area) * 100
        # Output the performance metrics
        performance_metrics = {
        "max_filled_area": max_filled_area,
        "min_filled_area": min_filled_area,
        "avg_filled_area": avg_filled_area,
        "trim_loss_percentage": trim_loss_percentage,
        "unused_stock_count": unused_stock_count,
        "trim_loss": trim_loss,
        "time_to_solve": None,
        }
        return performance_metrics  
    def aggregate_performance(self):
        """
        Aggregate the performance metrics from all test cases and format the output.

        Args:
        - self.performance: List of performance metrics dictionaries from all test cases.

        Returns:
        - Dictionary of aggregated and formatted metrics.
        """
        # Aggregate metrics across all test cases
        aggregated = {
            "max_filled_area": np.max([metrics["max_filled_area"] for metrics in self.performance]),
            "min_filled_area": np.min([metrics["min_filled_area"] for metrics in self.performance]),
            "avg_filled_area": np.mean([metrics["avg_filled_area"] for metrics in self.performance]),
            "max_time_to_solve": np.max([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            "min_time_to_solve": np.min([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            "avg_time_to_solve": np.mean([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            "avg_trim_loss_percentage": np.mean([metrics["trim_loss_percentage"] for metrics in self.performance]),
            "avg_unused_stock_count": np.mean([metrics["unused_stock_count"] for metrics in self.performance]),
            "total_trim_loss": np.sum([metrics["trim_loss"] for metrics in self.performance]),
            "total_unused_stocks": np.sum([metrics["unused_stock_count"] for metrics in self.performance]),
        }

        # Format the values to improve readability
        print(f"Max Filled Area: {aggregated['max_filled_area']:.2f}%")
        print(f"Min Filled Area: {aggregated['min_filled_area']:.2f}%")
        print(f"Avg Filled Area: {aggregated['avg_filled_area']:.2f}%")
        # print(f"Avg Trim Loss Percentage: {aggregated['avg_trim_loss_percentage']:.2f}%")
        print(f"Max Time To Solve: {aggregated['max_time_to_solve']:.4f}")
        print(f"Min Time To Solve: {aggregated['min_time_to_solve']:.4f}")
        print(f"Avg Time To Solve: {aggregated['avg_time_to_solve']:.4f}")
        print(f"Avg Unused Stock Count: {aggregated['avg_unused_stock_count']:.2f}%")
        # print(f"Total Trim Loss: {aggregated['total_trim_loss']:,}")
        # print(f"Total Unused Stocks: {aggregated['total_unused_stocks']:,}")


        
    