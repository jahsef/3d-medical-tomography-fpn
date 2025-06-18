import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

def gaussian_2d(x, y, cx, cy, sigma):
    """Create 2D gaussian centered at (cx, cy) with standard deviation sigma."""
    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

# Set up the figure and initial parameters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.2)

# Parameters
size = 64
initial_sigma = 2.0
center_x, center_y = size // 2, size // 2

# Create coordinate grids
x = np.arange(size)
y = np.arange(size)
X, Y = np.meshgrid(x, y)

# Initial gaussian blob
initial_blob = gaussian_2d(X, Y, center_x, center_y, initial_sigma)

# Plot 1: 2D heatmap
im1 = ax1.imshow(initial_blob, cmap='hot', interpolation='bilinear', origin='lower')
ax1.set_title('2D Gaussian Blob (Heatmap)')
ax1.set_xlabel('X pixels')
ax1.set_ylabel('Y pixels')
plt.colorbar(im1, ax=ax1, label='Intensity')

# Plot 2: 3D surface plot
ax2.remove()
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(X, Y, initial_blob, cmap='hot', alpha=0.8)
ax2.set_title('3D Gaussian Blob')
ax2.set_xlabel('X pixels')
ax2.set_ylabel('Y pixels')
ax2.set_zlabel('Intensity')
ax2.view_init(elev=30, azim=45)

# Add colorbar for 3D plot
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

# Create slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Sigma (σ)', 0.5, 32, valinit=initial_sigma, valfmt='%.1f')

def update(val):
    sigma = slider.val
    
    # Update gaussian blob
    blob = gaussian_2d(X, Y, center_x, center_y, sigma)
    
    # Update 2D heatmap
    im1.set_array(blob)
    im1.set_clim(0, blob.max())
    
    # Update 3D surface
    ax2.clear()
    surf = ax2.plot_surface(X, Y, blob, cmap='hot', alpha=0.8)
    ax2.set_title(f'3D Gaussian Blob (σ={sigma:.1f})')
    ax2.set_xlabel('X pixels')
    ax2.set_ylabel('Y pixels')
    ax2.set_zlabel('Intensity')
    ax2.view_init(elev=30, azim=45)
    
    fig.canvas.draw()

# Connect slider to update function
slider.on_changed(update)

# Add some text info
info_text = f"Patch size: {size}x{size}\nCenter: ({center_x}, {center_y})\nMax intensity at center"
fig.text(0.02, 0.95, info_text, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.show()

# Print some useful info
print(f"Patch dimensions: {size}x{size}")
print(f"Center position: ({center_x}, {center_y})")
print(f"Initial sigma: {initial_sigma}")
print(f"Max intensity (at center): {initial_blob.max():.3f}")
print(f"Intensity at 1σ distance: {np.exp(-0.5):.3f}")
print(f"Intensity at 2σ distance: {np.exp(-2):.3f}")
print(f"Intensity at 3σ distance: {np.exp(-4.5):.3f}")