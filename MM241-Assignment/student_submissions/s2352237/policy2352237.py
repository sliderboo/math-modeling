from policy import Policy

class Policy2352237(Policy):
    def __init__(self):
        self.skyline_parts = []
        self.used_stocks = 0
        self.sorted_stocks = []
        self.total_area_used = 0
        self.total_stock_area = 0

    def get_action(self, observation, info):
        if info['filled_ratio'] == 0.0:
            self.initialize_skyline_parts(observation["stocks"])

        products = sorted(observation["products"], key=lambda p: -(p["size"][0] * p["size"][1]))

        for product in products:
            if product["quantity"] == 0:
                continue

            product_size = sorted(product["size"], reverse=True)
            best_waste, best_position, best_stock_idx, rotate = float('inf'), (-1, -1), -1, False

            for i, (stock, stock_idx) in enumerate(self.sorted_stocks[:self.used_stocks]):
                waste, position, should_rotate = self.calculate_min_waste(product, stock, stock_idx)
                if waste < best_waste:
                    best_waste, best_position, best_stock_idx, rotate = waste, position, stock_idx, should_rotate

            while best_waste == float('inf'):
                self.used_stocks += 1
                new_stock = self.sorted_stocks[self.used_stocks - 1]
                stock_width, stock_height = self._get_stock_size_(new_stock[0])
                self.total_stock_area += stock_width * stock_height

                waste, position, should_rotate = self.calculate_min_waste(product, new_stock[0], new_stock[1])
                if waste < float('inf'):
                    best_waste, best_position, best_stock_idx, rotate = waste, position, new_stock[1], should_rotate

            if rotate:
                product["size"] = product["size"][::-1]

            self.place_product(product, best_position, observation["stocks"][best_stock_idx], best_stock_idx)
            self.total_area_used += product["size"][0] * product["size"][1]

            return {
                "stock_idx": best_stock_idx,
                "size": product["size"],
                "position": best_position
            }

    class SkylinePart:
        def __init__(self, start_x, end_x, height):
            self.start_x = start_x
            self.end_x = end_x
            self.height = height

        def intersects(self, product_size, position):
            product_start_x, product_y = position
            product_width, product_height = product_size
            product_end_x = product_start_x + product_width

            if product_end_x <= self.start_x or product_start_x >= self.end_x:
                return False

            if product_y >= self.height:
                return False

            return True

        def calculate_local_waste(self, product_size, position):
            product_start_x, product_y = position
            product_width, product_height = product_size
            product_end_x = product_start_x + product_width

            if product_end_x <= self.start_x or product_start_x >= self.end_x:
                return 0

            return (product_y - self.height) * (min(product_end_x, self.end_x) - max(product_start_x, self.start_x))
        

    def initialize_skyline_parts(self, stocks):
        self.skyline_parts = []
        self.sorted_stocks = sorted(
            [(stock, idx) for idx, stock in enumerate(stocks)],
            key=lambda s: -(self._get_stock_size_(s[0])[0] * self._get_stock_size_(s[0])[1])
        )

        for stock, idx in self.sorted_stocks:
            stock_width = self._get_stock_size_(stock)[0]
            self.skyline_parts.append([self.SkylinePart(0, stock_width, 0)])

    def calculate_min_waste(self, product, stock, stock_idx):
        stock_width, stock_height = self._get_stock_size_(stock)
        product_width, product_height = product["size"]

        min_waste = float('inf')
        best_position = (-1, -1)
        should_rotate = False

        for part in self.skyline_parts[stock_idx]:
            for x_offset in [part.start_x, part.end_x - product_width]:
                y_offset = part.height
                if x_offset < 0 or x_offset + product_width > stock_width or y_offset + product_height > stock_height:
                    continue

                if not self._can_place_(stock, (x_offset, y_offset), product["size"]):
                    continue

                if any(part.intersects(product["size"], (x_offset, y_offset)) for part in self.skyline_parts[stock_idx]):
                    continue

                waste = sum(part.calculate_local_waste(product["size"], (x_offset, y_offset)) for part in self.skyline_parts[stock_idx])
                if waste < min_waste:
                    min_waste, best_position, should_rotate = waste, (x_offset, y_offset), False

        product["size"] = product["size"][::-1]  # Rotate product

        for part in self.skyline_parts[stock_idx]:
            for x_offset in [part.start_x, part.end_x - product_width]:
                y_offset = part.height
                if x_offset < 0 or x_offset + product_width > stock_width or y_offset + product_height > stock_height:
                    continue

                if not self._can_place_(stock, (x_offset, y_offset), product["size"]):
                    continue

                if any(part.intersects(product["size"], (x_offset, y_offset)) for part in self.skyline_parts[stock_idx]):
                    continue

                waste = sum(part.calculate_local_waste(product["size"], (x_offset, y_offset)) for part in self.skyline_parts[stock_idx])
                if waste < min_waste:
                    min_waste, best_position, should_rotate = waste, (x_offset, y_offset), True

        product["size"] = product["size"][::-1]  # Restore original rotation
        return min_waste, best_position, should_rotate

    def place_product(self, product, position, stock, stock_idx):
        new_skyline = []
        product_width, product_height = product["size"]
        position_x, position_y = position
        x_start = position_x
        x_end = x_start + product_width
        height = position_y + product_height

        new_skyline.append(self.SkylinePart(x_start, x_end, height))

        for part in self.skyline_parts[stock_idx]:
            if part.end_x <= x_start or part.start_x >= x_end:
                new_skyline.append(part)
                continue

            if part.start_x < x_start:
                new_skyline.append(self.SkylinePart(part.start_x, x_start, part.height))

            if part.end_x > x_end:
                new_skyline.append(self.SkylinePart(x_end, part.end_x, part.height))

        self.skyline_parts[stock_idx] = sorted(new_skyline, key=lambda p: p.start_x)
