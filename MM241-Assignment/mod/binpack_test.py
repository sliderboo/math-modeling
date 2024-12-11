from rectpack import newPacker, PackingMode, PackingBin, MaxRectsBssf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def solve_2d_cutting_stock(rectangles, bins):
    """
    Solve the 2D Cutting Stock Problem using MaxRectsBinPacker BSSF.

    Args:
        rectangles (list of tuples): List of rectangles to pack (width, height).
        bins (list of tuples): List of available bins (width, height).
        
    Returns:
        list: List of bins with packed rectangles.
    """
    # Create a new packer instance
    packer = newPacker(mode=PackingMode.Offline, bin_algo=PackingBin.BBF, pack_algo=MaxRectsBssf, rotation=True)

    # Add the bins to the packer
    for b in bins:
        packer.add_bin(*b)

    # Add the rectangles to the packer
    for r in rectangles:
        packer.add_rect(*r)

    # Start packing
    packer.pack()

    # Collect packed results
    packed_bins = []
    for abin in packer:
        bin_rects = []
        for rect in abin:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            bin_rects.append({"x": x, "y": y, "width": w, "height": h})
        packed_bins.append(bin_rects)

    return packed_bins



def generate_data(n, m):
    prods = [(random.randint(25, 35), random.randint(25, 55)) for _ in range(n)]
    stocks = [(random.randint(200, 200), random.randint(300, 300)) for _ in range(m)]
    return stocks, prods

# Example input
# rectangles = [(100, 200), (150, 100), (50, 50), (80, 120), (10, 20), (35, 67), (13, 43), (25, 25)]  # Rectangles to pack (width, height)
# bins = [(250, 250)]  # Available bins (width, height)
bins, rectangles = generate_data(20000, 10)

print(rectangles)
print(bins)

# Solve the problem
packed_bins = solve_2d_cutting_stock(rectangles, bins)

# Output the packed results
for i, b in enumerate(packed_bins):
    print(f"Bin {i + 1}:")
    for rect in b:
        print(f"  Rectangle at ({rect['x']}, {rect['y']}), size ({rect['width']}x{rect['height']})")



def visualize_packed_bins(packed_bins, bins):
    for i, bin_rects in enumerate(packed_bins):
        bin_width, bin_height = bins[i]
        fig, ax = plt.subplots(1)
        ax.set_xlim(0, bin_width)
        ax.set_ylim(0, bin_height)
        ax.set_title(f'Bin {i + 1}')
        
        for rect in bin_rects:
            color = (random.random(), random.random(), random.random())
            rect_patch = patches.Rectangle((rect['x'], rect['y']), rect['width'], rect['height'], edgecolor='black', facecolor=color, alpha=0.5)
            ax.add_patch(rect_patch)
        
        plt.gca().invert_yaxis()
        plt.show()

# Visualize the packed bins
visualize_packed_bins(packed_bins, bins)