import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
import multiprocessing
from functools import lru_cache

@dataclass(frozen=True)
class Product:
    width: int
    height: int
    area: int = None
    is_balanced: bool = None

    def __post_init__(self):
        # Calculate derived attributes at initialization
        object.__setattr__(self, 'area', self.width * self.height)
        object.__setattr__(self, 'is_balanced', 
                          min(self.width, self.height) / max(self.width, self.height) > 0.8)

    @lru_cache(maxsize=1024)
    def get_orientations(self) -> List[Tuple[int, int]]:
        """Cache different orientations of the product"""
        if self.width != self.height:
            return [(self.width, self.height), (self.height, self.width)]
        return [(self.width, self.height)]

@dataclass
class Area:
    x: int
    y: int
    width: int
    height: int
    area: int = None

    def __post_init__(self):
        self.area = self.width * self.height

    def can_fit(self, product: Product) -> bool:
        return self.width >= product.width and self.height >= product.height

    def split_area(self, product: Product) -> List['Area']:
        """Optimized split area implementation using numpy for calculations"""
        new_areas = []
        
        if self.width > product.width and self.height > product.height:
            # Calculate areas for both split options at once
            areas = np.array([
                # Horizontal split areas
                [(self.width - product.width) * self.height,
                 product.width * (self.height - product.height)],
                # Vertical split areas
                [(self.width - product.width) * product.height,
                 self.width * (self.height - product.height)]
            ])
            
            # Use numpy for efficient comparison
            total_areas = np.sum(areas, axis=1)
            best_split = np.argmin(total_areas)
            
            if best_split == 0:  # Horizontal split
                if areas[0][0] > 0:
                    new_areas.append(Area(self.x + product.width, self.y, 
                                        self.width - product.width, self.height))
                if areas[0][1] > 0:
                    new_areas.append(Area(self.x, self.y + product.height,
                                        product.width, self.height - product.height))
            else:  # Vertical split
                if areas[1][0] > 0:
                    new_areas.append(Area(self.x + product.width, self.y,
                                        self.width - product.width, product.height))
                if areas[1][1] > 0:
                    new_areas.append(Area(self.x, self.y + product.height,
                                        self.width, self.height - product.height))
        elif self.width > product.width:
            new_areas.append(Area(self.x + product.width, self.y,
                                self.width - product.width, self.height))
        elif self.height > product.height:
            new_areas.append(Area(self.x, self.y + product.height,
                                self.width, self.height - product.height))
            
        return sorted(new_areas, key=lambda a: a.area, reverse=True)

class Stock:
    def __init__(self, stock_id: int, width: int, height: int):
        self.stock_id = stock_id
        self.width = width
        self.height = height
        self.area = width * height
        self.remaining_areas = deque([Area(0, 0, width, height)])
        self.placed_products = []
        self._lock = multiprocessing.Lock()

    def place_product(self, product: Product) -> bool:
        with self._lock:
            best_area = None
            best_waste = float('inf')
            best_orientation = None
            
            for width, height in product.get_orientations():
                for area in self.remaining_areas:
                    if area.width >= width and area.height >= height:
                        waste = (area.width - width) * (area.height - height)
                        if waste < best_waste:
                            best_waste = waste
                            best_area = area
                            best_orientation = (width, height)
            
            if best_area:
                self.placed_products.append((best_area.x, best_area.y, 
                                          best_orientation[0], best_orientation[1]))
                self.remaining_areas.remove(best_area)
                self.remaining_areas.extend(best_area.split_area(product))
                self._merge_areas()
                return True
            return False

    def _merge_areas(self):
        """Optimized area merging using numpy operations"""
        areas = np.array([(a.x, a.y, a.width, a.height) for a in self.remaining_areas])
        if len(areas) <= 1:
            return

        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(areas):
                j = i + 1
                while j < len(areas):
                    # Horizontal merge
                    if (areas[i][1] == areas[j][1] and 
                        areas[i][3] == areas[j][3] and 
                        areas[i][0] + areas[i][2] == areas[j][0]):
                        new_area = (areas[i][0], areas[i][1],
                                  areas[i][2] + areas[j][2], areas[i][3])
                        areas = np.delete(areas, j, 0)
                        areas[i] = new_area
                        merged = True
                        continue
                    
                    # Vertical merge
                    if (areas[i][0] == areas[j][0] and 
                        areas[i][2] == areas[j][2] and 
                        areas[i][1] + areas[i][3] == areas[j][1]):
                        new_area = (areas[i][0], areas[i][1],
                                  areas[i][2], areas[i][3] + areas[j][3])
                        areas = np.delete(areas, j, 0)
                        areas[i] = new_area
                        merged = True
                        continue
                    
                    j += 1
                i += 1
        
        self.remaining_areas = deque(Area(x, y, w, h) for x, y, w, h in areas)

def process_stock_batch(args):
    stock, products = args
    for product in products:
        if stock.place_product(product):
            continue
    return stock

def place_products_parallel(stocks: List[Stock], products: List[Product], 
                          batch_size: int = 1000):
    """Parallel implementation of product placement"""
    # Sort products by area in descending order
    products = sorted(products, key=lambda p: p.area, reverse=True)
    unplaced_products = products.copy()
    
    # Calculate number of threads based on CPU cores
    num_threads = multiprocessing.cpu_count()
    
    while unplaced_products:
        # Process stocks in parallel batches
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create batches of products for each stock
            stock_batches = [
                (stock, unplaced_products[:batch_size]) 
                for stock in stocks
            ]
            
            # Process batches in parallel
            futures = [
                executor.submit(process_stock_batch, batch) 
                for batch in stock_batches
            ]
            
            # Collect results
            processed_stocks = [
                future.result() 
                for future in as_completed(futures)
            ]
            
            # Update unplaced products
            placed_count = sum(len(stock.placed_products) for stock in processed_stocks)
            unplaced_products = unplaced_products[placed_count:]
            
            if not placed_count:
                print(f"Warning: Could not place {len(unplaced_products)} remaining products")
                break

def generate_random_stock(stock_id: int) -> Stock:
    """Optimized random stock generation"""
    width = np.random.randint(1000, 5000)
    height = np.random.randint(1000, 5000)
    return Stock(stock_id, width, height)

def generate_random_product() -> Product:
    """Optimized random product generation"""
    width = np.random.randint(100, 500)
    height = np.random.randint(100, 500)
    return Product(width, height)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate stocks and products using numpy's faster random number generation
    num_stocks = 10000
    num_products = 100000
    
    # Parallel generation of stocks and products
    with ThreadPoolExecutor() as executor:
        stocks = list(executor.map(generate_random_stock, range(num_stocks)))
        products = [generate_random_product() for _ in range(num_products)]
    
    # Place products using parallel implementation
    place_products_parallel(stocks, products)
    print("Done")