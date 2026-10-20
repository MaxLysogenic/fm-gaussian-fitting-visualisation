from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

from photon_data import GRID_SIZE, PIXEL_SIZE_NM, load_camera_data

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
MAGNIFICATION = 150
PHYSICAL_PIXEL_UM = (PIXEL_SIZE_NM * MAGNIFICATION) / 1000.0

all_photons_x, all_photons_y, photon_counts, digitized_grid = load_camera_data()
fit_pixels = np.arange(GRID_SIZE) + 0.5
X_grid, Y_grid = np.meshgrid(fit_pixels, fit_pixels)

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
    "Camera Digitization",
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
        rect = Rectangle(
            (i, j),
            1,
            1,
            fill=False,
            edgecolor='#30363d',
            lw=1.5,
            zorder=3,
        )
        ax.add_patch(rect)

# Retain the highlighted pixel and scale bar geometry from the previous phase.
sample_pixel = Rectangle(
    (GRID_SIZE - 1, 0),
    1,
    1,
    fill=True,
    facecolor='#21262d',
    edgecolor='#58a6ff',
    lw=2,
    zorder=4,
)
sample_pixel.set_alpha(0.0)
ax.add_patch(sample_pixel)

scale_bar = Rectangle(
    (GRID_SIZE - 1, -0.3),
    1.0,
    0.08,
    facecolor='#58a6ff',
    edgecolor='none',
    zorder=5,
)
ax.add_patch(scale_bar)

scale_label = ax.text(
    0.0,
    -0.5,
    "Effective Pixel Size:",
    ha='left',
    va='top',
    fontsize=20,
    color='#e6edf3',
    fontweight='bold',
)
scale_value = ax.text(
    5.2,
    -0.5,
    f"{PIXEL_SIZE_NM:.0f}",
    ha='right',
    va='top',
    fontsize=20,
    color='#e6edf3',
    fontweight='bold',
)
scale_unit = ax.text(
    GRID_SIZE,
    -0.5,
    "nm",
    ha='right',
    va='top',
    fontsize=20,
    color='#e6edf3',
    fontweight='bold',
)

pixel_counts_2d = digitized_grid
pixel_counts_image = ax.imshow(
    pixel_counts_2d,
    cmap='viridis',
    interpolation='nearest',
    origin='lower',
    extent=[0, GRID_SIZE, 0, GRID_SIZE],
    alpha=0.0,
    zorder=2,
)

photon_points, = ax.plot(
    all_photons_x,
    all_photons_y,
    marker='o',
    linestyle='None',
    markersize=2.5,
    color='#00ffcc',
    alpha=0.45,
    zorder=5,
)

count_texts = []
for j in range(GRID_SIZE):
    for i in range(GRID_SIZE):
        count_text = ax.text(
            i + 0.5,
            j + 0.5,
            str(photon_counts[j, i]),
            ha='center',
            va='center',
            fontsize=11,
            color='white',
            fontweight='bold',
            alpha=0.0,
            zorder=4.5,
        )
        count_text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='#0d1117'),
            path_effects.Normal(),
        ])
        count_texts.append(count_text)

# ==========================================
# 2. ANIMATION LOGIC
# ==========================================
FPS = 20
TRANSITION_FRAMES = 20
ANIMATION_FRAMES = TRANSITION_FRAMES

def init():
    photon_points.set_alpha(0.45)
    pixel_counts_image.set_alpha(0.0)
    for count_text in count_texts:
        count_text.set_alpha(0.0)
    return [photon_points, pixel_counts_image, *count_texts, sample_pixel, scale_bar, scale_label, scale_value, scale_unit, title_text]

def animate(frame):
    progress = np.clip(frame / float(TRANSITION_FRAMES - 1), 0.0, 1.0)
    photon_points.set_alpha(0.45 * (1.0 - progress))
    pixel_counts_image.set_alpha(progress)
    for count_text in count_texts:
        count_text.set_alpha(progress)
    return [photon_points, pixel_counts_image, *count_texts, sample_pixel, scale_bar, scale_label, scale_value, scale_unit, title_text]

# ==========================================
# 3. RENDER & SAVE TO MP4
# ==========================================
if __name__ == "__main__":
    print("Rendering Phase 3: Camera Digitization...")
    print(
        f"Binned {photon_counts.sum()} of {all_photons_x.size} photons "
        f"into a {GRID_SIZE}x{GRID_SIZE} camera."
    )
    print("Per-pixel photon counts (rows=y from bottom to top, columns=x left to right):")
    print(photon_counts)
    output_directory = Path(__file__).resolve().parent / "generated_videos"
    output_directory.mkdir(exist_ok=True)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=ANIMATION_FRAMES,
        init_func=init,
        interval=1000 / FPS,
        blit=True,
    )

    output_filename = output_directory / "smt_3.mp4"
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata=dict(artist='Max Lian'),
        bitrate=1800,
    )

    anim.save(output_filename, writer=writer)
    plt.close()
    print(f"Done! Saved: {output_filename}")
