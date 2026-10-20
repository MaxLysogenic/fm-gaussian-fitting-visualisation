from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

from photon_data import (
    GRID_SIZE,
    PIXEL_SIZE_NM,
    fit_gaussian,
    load_camera_data,
)

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
MAGNIFICATION = 150
PHYSICAL_PIXEL_UM = (PIXEL_SIZE_NM * MAGNIFICATION) / 1000.0

LOCALIZATION_RADIUS_PX = 30 / PIXEL_SIZE_NM

all_photons_x, all_photons_y, photon_counts, digitized_grid = load_camera_data()
fit_parameters, X_grid, Y_grid, fitted_pixels = fit_gaussian(digitized_grid)
amplitude_fit, x_fit, y_fit, sigma_fit, offset_fit = fit_parameters

SURFACE_RESOLUTION = 200
surface_axis = np.linspace(0, GRID_SIZE, SURFACE_RESOLUTION)
surface_x_grid, surface_y_grid = np.meshgrid(surface_axis, surface_axis)
fitted_surface = (
    offset_fit
    + amplitude_fit
    * np.exp(
        -((surface_x_grid - x_fit) ** 2 + (surface_y_grid - y_fit) ** 2)
        / (2 * sigma_fit**2)
    )
)

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
    "Sub-Pixel Localisation",
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

# Draw the same 6x6 wireframe grid.
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

# Cross-fade from the fitted surface back to the camera's digitized grid.
surface_vmin = 0.0
surface_vmax = max(float(np.max(digitized_grid)), float(np.max(fitted_pixels)))
pixel_counts_image = ax.imshow(
    digitized_grid,
    extent=[0, GRID_SIZE, 0, GRID_SIZE],
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=surface_vmin,
    vmax=surface_vmax,
    alpha=0.0,
    zorder=2,
)
smooth_psf_image = ax.imshow(
    fitted_surface,
    extent=[0, GRID_SIZE, 0, GRID_SIZE],
    origin='lower',
    cmap='viridis',
    interpolation='bilinear',
    vmin=surface_vmin,
    vmax=surface_vmax,
    alpha=1.0,
    zorder=2.1,
)

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

count_texts = []
for j in range(GRID_SIZE):
    for i in range(GRID_SIZE):
        count_text = ax.text(
            i + 0.5,
            j + 0.5,
            str(int(np.rint(digitized_grid[j, i]))),
            ha='center',
            va='center',
            fontsize=11,
            color='white',
            fontweight='bold',
            alpha=1.0,
            zorder=4.5,
        )
        count_text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='#0d1117'),
            path_effects.Normal(),
        ])
        count_texts.append(count_text)

# Match the fitted Gaussian contour levels shown in Phase 4.
minimum_grid_fraction = float(
    np.min((fitted_surface - offset_fit) / amplitude_fit)
)
CONTOUR_FRACTIONS = np.concatenate(
    (
        np.geomspace(minimum_grid_fraction, 0.1, 10),
        np.linspace(0.2, 0.99, 9),
    )
)
INITIAL_CONTOUR_SCALES = np.sqrt(-2 * np.log(CONTOUR_FRACTIONS))
CIRCLE_ANGLES = np.linspace(0, 2 * np.pi, 361)

grid_clip = Rectangle(
    (0, 0),
    GRID_SIZE,
    GRID_SIZE,
    transform=ax.transData,
)
ring_lines = []
for contour_scale in INITIAL_CONTOUR_SCALES:
    radius = sigma_fit * contour_scale
    x_offset = radius * np.cos(CIRCLE_ANGLES)
    y_offset = radius * np.sin(CIRCLE_ANGLES)
    line, = ax.plot(
        x_fit + x_offset,
        y_fit + y_offset,
        color='#00ffff',
        linewidth=1.4,
        alpha=1.0,
        zorder=4,
    )
    line.set_clip_path(grid_clip)
    ring_lines.append(line)

centroid_marker, = ax.plot(
    [x_fit],
    [y_fit],
    marker='+',
    linestyle='None',
    markersize=10,
    markeredgewidth=2,
    color='white',
    alpha=0.0,
    zorder=10,
)

converged_circle_lines = []
for linewidth, alpha in ((7.0, 0.12), (4.0, 0.28), (2.0, 1.0)):
    circle, = ax.plot(
        x_fit + LOCALIZATION_RADIUS_PX * np.cos(CIRCLE_ANGLES),
        y_fit + LOCALIZATION_RADIUS_PX * np.sin(CIRCLE_ANGLES),
        color='#00ffcc',
        linewidth=linewidth,
        linestyle=(0, (5, 3)),
        alpha=alpha,
        zorder=7,
    )
    converged_circle_lines.append(circle)

annotation_effects = [
    path_effects.Stroke(linewidth=2.5, foreground='#0d1117'),
    path_effects.Normal(),
]
centroid_text = ax.text(
    GRID_SIZE / 2,
    5.55,
    f"Inferred Fluorophore Position",
    ha='center',
    va='center',
    fontsize=20,
    color='#00ffcc',
    fontweight='bold',
    alpha=0.0,
    zorder=10,
)
centroid_text.set_path_effects(annotation_effects)
uncertainty_text = ax.text(
    GRID_SIZE / 2,
    5.15,
    "Localisation Precision: ±30 nm",
    ha='center',
    va='center',
    fontsize=18,
    color="#fdffff",
    fontweight='bold',
    alpha=0.0,
    zorder=10,
)
uncertainty_text.set_path_effects(annotation_effects)
arrow_start = (GRID_SIZE / 2, 4.95)
arrow_direction = np.array(arrow_start) - np.array((x_fit, y_fit))
arrow_length = np.linalg.norm(arrow_direction)
arrow_endpoint = (
    x_fit + LOCALIZATION_RADIUS_PX * arrow_direction[0] / arrow_length,
    y_fit + LOCALIZATION_RADIUS_PX * arrow_direction[1] / arrow_length,
)
precision_arrow = ax.annotate(
    "",
    xy=arrow_endpoint,
    xytext=arrow_start,
    arrowprops={
        "arrowstyle": "-|>",
        "color": "#00ffcc",
        "linewidth": 2.2,
        "shrinkA": 0,
        "shrinkB": 0,
    },
    zorder=6,
)
precision_arrow.set_alpha(0.0)
precision_arrow.arrow_patch.set_path_effects([
    path_effects.Stroke(linewidth=4, foreground='#0d1117'),
    path_effects.Normal(),
])

# ==========================================
# 2. ANIMATION LOGIC
# ==========================================
FPS = 20
SURFACE_TRANSITION_END = 20
PIXEL_HEATMAP_MAX_ALPHA = 0.3
CONVERGENCE_START = 15
CONVERGENCE_END = 40
CENTROID_FADE_START = 35
CENTROID_FADE_END = 50
HOLD_FRAMES = 10
ANIMATION_END_FRAME = CENTROID_FADE_END + HOLD_FRAMES
ANIMATION_FRAMES = ANIMATION_END_FRAME + 1

def init():
    pixel_counts_image.set_alpha(0.0)
    smooth_psf_image.set_alpha(1.0)
    for text in count_texts:
        text.set_alpha(0.0)
    for line in ring_lines:
        line.set_alpha(1.0)
    for line in converged_circle_lines:
        line.set_alpha(0.0)
    centroid_text.set_alpha(0.0)
    uncertainty_text.set_alpha(0.0)
    centroid_marker.set_alpha(0.0)
    precision_arrow.arrow_patch.set_alpha(0.0)
    return [
        pixel_counts_image,
        smooth_psf_image,
        *count_texts,
        *ring_lines,
        *converged_circle_lines,
        centroid_marker,
        centroid_text,
        uncertainty_text,
        precision_arrow,
        sample_pixel,
        scale_bar,
        scale_label,
        scale_value,
        scale_unit,
        title_text,
    ]

def animate(frame):
    surface_progress = np.clip(frame / SURFACE_TRANSITION_END, 0.0, 1.0)
    pixel_counts_image.set_alpha(PIXEL_HEATMAP_MAX_ALPHA * surface_progress)

    convergence_progress = np.clip(
        (frame - CONVERGENCE_START) / (CONVERGENCE_END - CONVERGENCE_START),
        0.0,
        1.0,
    )
    ring_scale = 1.0 - convergence_progress
    smooth_psf_image.set_alpha(1.0 - convergence_progress)
    
    for line, contour_scale in zip(ring_lines, INITIAL_CONTOUR_SCALES):
        radius = sigma_fit * contour_scale * ring_scale
        ellipse_x = radius * np.cos(CIRCLE_ANGLES)
        ellipse_y = radius * np.sin(CIRCLE_ANGLES)
        line.set_data(
            x_fit + ellipse_x,
            y_fit + ellipse_y,
        )
        line.set_alpha(1.0 - convergence_progress)

    circle_progress = convergence_progress
    for line, final_alpha in zip(converged_circle_lines, (0.12, 0.28, 1.0)):
        line.set_alpha(final_alpha * circle_progress)

    centroid_progress = np.clip(
        (frame - CENTROID_FADE_START) / (CENTROID_FADE_END - CENTROID_FADE_START),
        0.0,
        1.0,
    )
    centroid_text.set_alpha(centroid_progress)
    uncertainty_text.set_alpha(centroid_progress)
    centroid_marker.set_alpha(centroid_progress)
    precision_arrow.arrow_patch.set_alpha(centroid_progress)

    return [
        pixel_counts_image,
        smooth_psf_image,
        *count_texts,
        *ring_lines,
        *converged_circle_lines,
        centroid_marker,
        centroid_text,
        uncertainty_text,
        precision_arrow,
        sample_pixel,
        scale_bar,
        scale_label,
        scale_value,
        scale_unit,
        title_text,
    ]

# ==========================================
# 3. RENDER & SAVE TO MP4
# ==========================================
if __name__ == "__main__":
    print("Rendering Phase 5: Sub-Pixel Localisation...")
    print(f"Fitted Gaussian centroid: ({x_fit:.3f}, {y_fit:.3f}) px")
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

    output_filename = output_directory / "smt_5.mp4"
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata=dict(artist='Max Lian'),
        bitrate=1800,
    )

    anim.save(output_filename, writer=writer)
    plt.close()
    print(f"Done! Saved: {output_filename}")
