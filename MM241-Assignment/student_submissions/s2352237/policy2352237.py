from policy import Policy


class Policy2352237(Policy):
    def __init__(self):
        self.height_areas = []
        self.current_stock = 0
        self.stock_inventory = []
        self.total_prod_area = 0
        self.total_stock_area = 0

    def get_action(self, observation, info):
        if info['filled_ratio'] == 0.0:
            self.initialize_height_areas(observation["stocks"])

        prods = sorted(observation["products"], key=lambda p: -(p["size"][0] * p["size"][1]))

        for prod in prods:
            if prod["quantity"] == 0:
                continue

            prod_size = sorted(prod["size"], reverse=True)
            min_waste, best_position, selected_stock, rotate = float('inf'), (-1, -1), -1, False

            for i, (stock, stock_idx) in enumerate(self.stock_inventory[:self.current_stock]):
                waste, position, should_rotate = self.best_placement(prod, stock, stock_idx)
                if waste < min_waste:
                    min_waste, best_position, selected_stock, rotate = waste, position, stock_idx, should_rotate

            while min_waste == float('inf'):
                self.current_stock += 1
                new_stock = self.stock_inventory[self.current_stock - 1]
                stock_width, stock_height = self._get_stock_size_(new_stock[0])
                self.total_stock_area += stock_width * stock_height

                waste, position, should_rotate = self.best_placement(prod, new_stock[0], new_stock[1])
                if waste < float('inf'):
                    min_waste, best_position, selected_stock, rotate = waste, position, new_stock[1], should_rotate

            if rotate:
                prod["size"] = prod["size"][::-1]

            self.update_height_area(prod, best_position, observation["stocks"][selected_stock], selected_stock)
            self.total_prod_area += prod["size"][0] * prod["size"][1]

            return {
                "stock_idx": selected_stock,
                "size": prod["size"],
                "position": best_position
            }

    class HeightSegment:
        def __init__(self, start_x, end_x, level):
            self.start_x = start_x
            self.end_x = end_x
            self.level = level

        def has_overlap(self, prod_size, position):
            prod_start_x, prod_y = position
            prod_width, prod_height = prod_size
            prod_end_x = prod_start_x + prod_width

            if prod_end_x <= self.start_x or prod_start_x >= self.end_x:
                return False

            if prod_y >= self.level:
                return False

            return True

        def compute_gap_area(self, prod_size, position):
            prod_start_x, prod_y = position
            prod_width, prod_height = prod_size
            prod_end_x = prod_start_x + prod_width

            if prod_end_x <= self.start_x or prod_start_x >= self.end_x:
                return 0

            return (prod_y - self.level) * (min(prod_end_x, self.end_x) - max(prod_start_x, self.start_x))

    def initialize_height_areas(self, stocks):
        self.height_areas = []
        self.stock_inventory = sorted(
            [(stock, idx) for idx, stock in enumerate(stocks)],
            key=lambda s: -(self._get_stock_size_(s[0])[0] * self._get_stock_size_(s[0])[1])
        )

        for stock, idx in self.stock_inventory:
            stock_width = self._get_stock_size_(stock)[0]
            self.height_areas.append([self.HeightSegment(0, stock_width, 0)])

    def best_placement(self, prod, stock, stock_idx):
        stock_width, stock_height = self._get_stock_size_(stock)
        prod_width, prod_height = prod["size"]

        min_waste = float('inf')
        best_position = (-1, -1)
        should_rotate = False

        for segment in self.height_areas[stock_idx]:
            for x_offset in [segment.start_x, segment.end_x - prod_width]:
                y_offset = segment.level
                if x_offset < 0 or x_offset + prod_width > stock_width or y_offset + prod_height > stock_height:
                    continue

                if not self._can_place_(stock, (x_offset, y_offset), prod["size"]):
                    continue

                if any(segment.has_overlap(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx]):
                    continue

                waste = sum(segment.compute_gap_area(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx])
                if waste < min_waste:
                    min_waste, best_position, should_rotate = waste, (x_offset, y_offset), False

        prod["size"] = prod["size"][::-1]  # Rotate prod

        for segment in self.height_areas[stock_idx]:
            for x_offset in [segment.start_x, segment.end_x - prod_width]:
                y_offset = segment.level
                if x_offset < 0 or x_offset + prod_width > stock_width or y_offset + prod_height > stock_height:
                    continue

                if not self._can_place_(stock, (x_offset, y_offset), prod["size"]):
                    continue

                if any(segment.has_overlap(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx]):
                    continue

                waste = sum(segment.compute_gap_area(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx])
                if waste < min_waste:
                    min_waste, best_position, should_rotate = waste, (x_offset, y_offset), True

        prod["size"] = prod["size"][::-1]  # Restore original rotation
        return min_waste, best_position, should_rotate

    def update_height_area(self, prod, position, stock, stock_idx):
        updated_area = []
        prod_width, prod_height = prod["size"]
        position_x, position_y = position
        start = position_x
        end = start + prod_width
        new_height = position_y + prod_height

        updated_area.append(self.HeightSegment(start, end, new_height))

        for segment in self.height_areas[stock_idx]:
            if segment.end_x <= start or segment.start_x >= end:
                updated_area.append(segment)
                continue

            if segment.start_x < start:
                updated_area.append(self.HeightSegment(segment.start_x, start, segment.level))

            if segment.end_x > end:
                updated_area.append(self.HeightSegment(end, segment.end_x, segment.level))

        self.height_areas[stock_idx] = sorted(updated_area, key=lambda p: p.start_x)