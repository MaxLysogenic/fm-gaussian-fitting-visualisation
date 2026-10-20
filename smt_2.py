import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

from photon_data import PHOTON_FILE, PSF_SIGMA, PSF_X0, PSF_Y0

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
GRID_SIZE = 6               # 6x6 pixel grid
PIXEL_SIZE_NM = 107.0       # Effective pixel size
MAGNIFICATION = 150         # Objective magnification
PHYSICAL_PIXEL_UM = (PIXEL_SIZE_NM * MAGNIFICATION) / 1000.0  # ~16.05 um

# Canvas styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Set layout bounds
ax.set_xlim(-0.8, GRID_SIZE + 0.8)
ax.set_ylim(-1.0, GRID_SIZE + 0.45)
ax.set_aspect('equal')
ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

# Header & Subtitle text elements
title_text = ax.text(
    GRID_SIZE / 2,
    GRID_SIZE + 0.2,
    "Photon Distribution Simulation",
    ha='center',
    va='center',
    fontsize=20,
    fontweight='bold',
    color='#ffffff',
    zorder=8,
)
title_text.set_path_effects([
    path_effects.Stroke(linewidth=3, foreground='#0d1117'),
    path_effects.Normal(),
])

# Draw 6x6 Wireframe Grid
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='#30363d', lw=1.5, zorder=3)
        ax.add_patch(rect)

np.random.seed(42)
signal_photons = np.empty((0, 2))
while signal_photons.shape[0] < 2000:
    candidates = np.random.normal(
        loc=(PSF_X0, PSF_Y0),
        scale=PSF_SIGMA,
        size=(2000 - signal_photons.shape[0], 2),
    )
    in_grid = np.all((candidates >= 0.0) & (candidates <= GRID_SIZE), axis=1)
    signal_photons = np.vstack([signal_photons, candidates[in_grid]])
photons_x, photons_y = signal_photons.T
bg_x = np.random.uniform(low=0.0, high=float(GRID_SIZE), size=350)
bg_y = np.random.uniform(low=0.0, high=float(GRID_SIZE), size=350)
all_photons_x = np.hstack([photons_x, bg_x])
all_photons_y = np.hstack([photons_y, bg_y])

photon_points, = ax.plot(
    [],
    [],
    marker='o',
    linestyle='None',
    markersize=2.5,
    color='#00ffcc',
    alpha=0.45,
    zorder=5,
)

# Highlight single pixel frame (bottom-right cell for scale bar reference)
sample_pixel = Rectangle((GRID_SIZE - 1, 0), 1, 1, fill=True, facecolor='#21262d', edgecolor='#58a6ff', lw=2, zorder=4)
ax.add_patch(sample_pixel)

# Scale Bar UI Elements
scale_bar = Rectangle((GRID_SIZE - 1, -0.3), 1.0, 0.08, facecolor='#58a6ff', edgecolor='none', zorder=5)
ax.add_patch(scale_bar)

scale_label = ax.text(0.0, -0.5, "Physical Pixel Size:", ha='left', va='top', fontsize=20, color='#e6edf3', fontweight='bold')
scale_value = ax.text(5.2, -0.5, "16", ha='right', va='top', fontsize=20, color='#e6edf3', fontweight='bold')
scale_unit = ax.text(GRID_SIZE, -0.5, "µm", ha='right', va='top', fontsize=20, color='#e6edf3', fontweight='bold')
scale_bar.set_facecolor('#3fb950')  # Green highlight

# ==========================================
# 2. ANIMATION LOGIC
# ==========================================
FPS = 20
ANIMATION_FRAMES = 30

def init():
    scale_label.set_text("Effective Pixel Size:")
    scale_value.set_text(f"{PIXEL_SIZE_NM:.0f}")
    scale_unit.set_text("nm")
    sample_pixel.set_alpha(1.0)
    scale_bar.set_facecolor('#58a6ff')
    photon_points.set_data([], [])
    return [sample_pixel, photon_points, scale_bar, scale_label, scale_value, scale_unit, title_text]

def animate(frame):
    photons_per_frame = int(np.ceil(all_photons_x.size / ANIMATION_FRAMES))
    visible_count = min((frame + 1) * photons_per_frame, all_photons_x.size)
    photon_points.set_data(
        all_photons_x[:visible_count],
        all_photons_y[:visible_count],
    )
    sample_pixel.set_alpha(0.0)
    return [sample_pixel, photon_points, scale_bar, scale_label, scale_value, scale_unit, title_text]

# ==========================================
# 3. RENDER & SAVE TO MP4
# ==========================================
if __name__ == "__main__":
    print("Rendering Phase 2: Signal Photons Hit Detector...")
    np.savez(
        PHOTON_FILE,
        photons_x=all_photons_x,
        photons_y=all_photons_y,
        psf_center=(PSF_X0, PSF_Y0),
    )
    print(f"Saved {all_photons_x.size} photon arrivals to {PHOTON_FILE}")
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=ANIMATION_FRAMES,
        init_func=init,
        interval=1000 / FPS,
        blit=True,
    )

    output_filename = "smt_2.mp4"
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Max Lian'), bitrate=1800)

    anim.save(output_filename, writer=writer)
    plt.close()
    print(f"Done! Saved: {output_filename}")
