from heapq import merge
import matplotlib.pyplot as plt
import random

class Rectangle:
    def __init__(self, x, y, width, height, rid=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rid = rid

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

class HSegment:
    def __init__(self, left, top, length):
        self.left = left
        self.top = top
        self.length = length

    @property
    def right(self):
        return self.left + self.length

class SkylineSolver:
    def __init__(self, bin_width, bin_height, allow_rotation=True):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.allow_rotation = allow_rotation
        self._skyline = [HSegment(0, 0, bin_width)]
        self.placed_rectangles = []

    def _placement_points(self, width):
        skyline_r = self._skyline[-1].right
        skyline_l = self._skyline[0].left
        
        points_left = (s.left for s in self._skyline if s.left + width <= skyline_r)
        points_right = (s.right - width for s in self._skyline if s.right - width >= skyline_l)
        
        return merge(points_left, points_right)

    def _generate_placements(self, width, height):
        points = []
        left_idx = 0
        right_idx = 0

        for p in self._placement_points(width):
            while right_idx < len(self._skyline) and p + width > self._skyline[right_idx].right:
                right_idx += 1

            while left_idx < len(self._skyline) and p >= self._skyline[left_idx].right:
                left_idx += 1

            if left_idx >= len(self._skyline) or right_idx >= len(self._skyline):
                break

            max_top = max(seg.top for seg in self._skyline[left_idx:right_idx + 1])

            if max_top + height <= self.bin_height:
                points.append((Rectangle(p, max_top, width, height), left_idx, right_idx))

        return points

    def _merge_skyline(self):
        merged = [self._skyline[0]]
        for seg in self._skyline[1:]:
            last = merged[-1]
            if seg.top == last.top:
                merged[-1] = HSegment(last.left, last.top, last.length + seg.length)
            else:
                merged.append(seg)
        self._skyline = merged

    def _add_to_skyline(self, rect):
        new_skyline = []
        for seg in self._skyline:
            if seg.right <= rect.x or seg.left >= rect.right:
                new_skyline.append(seg)
            else:
                if seg.left < rect.x:
                    new_skyline.append(HSegment(seg.left, seg.top, rect.x - seg.left))
                if seg.right > rect.right:
                    new_skyline.append(HSegment(rect.right, seg.top, seg.right - rect.right))

        new_skyline.append(HSegment(rect.x, rect.bottom, rect.width))
        self._skyline = sorted(new_skyline, key=lambda s: s.left)
        self._merge_skyline()

    def _select_position(self, width, height):
        placements = self._generate_placements(width, height)
        if self.allow_rotation and width != height:
            placements += self._generate_placements(height, width)

        if not placements:
            return None, None

        best = min(placements, key=lambda p: (p[0].y, p[0].x))
        return best[0], best[1:]

    def add_rectangle(self, width, height, rid=None):
        rect, _ = self._select_position(width, height)
        if not rect:
            return None

        self._add_to_skyline(rect)
        rect.rid = rid
        self.placed_rectangles.append(rect)
        return rect

    def reset(self):
        self._skyline = [HSegment(0, 0, self.bin_width)]
        self.placed_rectangles = []

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.bin_width)
        ax.set_ylim(0, self.bin_height)
        ax.set_aspect('equal', adjustable='box')
        
        for rect in self.placed_rectangles:
            ax.add_patch(plt.Rectangle((rect.x, rect.y), rect.width, rect.height, edgecolor="black"))

        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.title("Skyline Packing Visualization")
        plt.show()

def generate_data(n, m):
    prods = [(random.randint(25, 35), random.randint(25, 55)) for _ in range(n)]
    stocks = [(random.randint(200, 200), random.randint(300, 300)) for _ in range(m)]
    return prods, stocks


# Example input

# Example usage
if __name__ == "__main__":
    prods, stocks = generate_data(50, 1)
    print(prods)
    print(stocks)
    solver = SkylineSolver(stocks[0][0], stocks[0][1], allow_rotation=True)

    for i, p in enumerate(prods):
        rect = solver.add_rectangle(*p, rid=i)


    for r in solver.placed_rectangles:
        print(f"Rectangle ID {r.rid}: Placed at ({r.x}, {r.y}) with width {r.width} and height {r.height}")

    solver.visualize()