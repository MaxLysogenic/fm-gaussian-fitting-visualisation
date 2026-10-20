import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch, Rectangle

# ==========================================
# 1. PARAMETERS & GEOMETRY
# ==========================================
GRID_SIZE = 6               # 6x6 pixel grid
PIXEL_SIZE_NM = 107.0        # Effective pixel size
TARGET_MAGNIFICATION = 150.0
PHYSICAL_PIXEL_NM = 16050.0  # 107 nm * 150
PHYSICAL_PIXEL_UM = PHYSICAL_PIXEL_NM / 1000.0

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
    "Pixel Size Scaling",
    ha='center',
    va='center',
    fontsize=20,
    fontweight='bold',
    color='#ffffff',
    zorder=8,
)

# Draw 6x6 Wireframe Grid
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        rect = Rectangle((i, j), 1, 1, fill=False, edgecolor='#30363d', lw=1.5, zorder=3)
        ax.add_patch(rect)

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

annotation_effects = [
    path_effects.Stroke(linewidth=3, foreground='#0d1117'),
    path_effects.Normal(),
]
title_text.set_path_effects(annotation_effects)
magnification_text = ax.text(
    GRID_SIZE / 2,
    GRID_SIZE / 2,
    "",
    ha='center',
    va='center',
    fontsize=20,
    fontweight='bold',
    color='#ffffff',
    zorder=8,
)
magnification_text.set_path_effects(annotation_effects)

arrow_starts = [
    (1.7, 3.0),
    (4.3, 3.0),
    (3.0, 2.4),
    (3.0, 3.6),
]
grid_edges = [(0, 3), (6, 3), (3, 0), (3, 6)]
zoom_arrows = []
for start, end in zip(arrow_starts, grid_edges):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=4,
        color='#ffffff',
        zorder=7,
        clip_on=True,
    )
    arrow.set_path_effects([
        path_effects.Stroke(linewidth=4, foreground='#0d1117'),
        path_effects.Normal(),
    ])
    ax.add_patch(arrow)
    zoom_arrows.append(arrow)

# ==========================================
# 2. ANIMATION LOGIC (PHASE 1 ONLY)
# ==========================================
FPS = 20
TOTAL_FRAMES = 45
ARROW_FADE_FRAMES = FPS
ANIMATION_FRAMES = TOTAL_FRAMES + ARROW_FADE_FRAMES

def init():
    scale_label.set_text("Physical Pixel Size:")
    scale_value.set_text(f"{PHYSICAL_PIXEL_UM:.0f}")
    scale_unit.set_text("µm")
    scale_bar.set_facecolor('#3fb950')
    magnification_text.set_text("M: 1.0x")
    for arrow in zoom_arrows:
        arrow.set_alpha(1.0)
    return [
        scale_bar,
        scale_label,
        scale_value,
        scale_unit,
        magnification_text,
        *zoom_arrows,
        title_text,
    ]

def animate(frame):
    if frame == 0:
        return init()

    zoom_frame = min(frame, TOTAL_FRAMES - 1)
    progress = zoom_frame / float(TOTAL_FRAMES - 1)

    current_magnification = 1.0 + progress * (TARGET_MAGNIFICATION - 1.0)
    effective_nm = PHYSICAL_PIXEL_NM / current_magnification
    magnification_text.set_text(f"M: {current_magnification:.1f}x")
    scale_label.set_text("Effective Pixel Size:")
    if effective_nm > 1000:
        scale_value.set_text(f"{effective_nm / 1000.0:.2f}")
        scale_unit.set_text("µm")
    else:
        scale_value.set_text(f"{effective_nm:.0f}")
        scale_unit.set_text("nm")
    scale_bar.set_facecolor('#58a6ff')  # Blue highlight

    arrow_fade_progress = np.clip(
        (frame - (TOTAL_FRAMES - 1)) / ARROW_FADE_FRAMES,
        0.0,
        1.0,
    )
    for arrow in zoom_arrows:
        arrow.set_alpha(1.0 - arrow_fade_progress)
    magnification_text.set_alpha(1.0 - arrow_fade_progress)

    return [
        scale_bar,
        scale_label,
        scale_value,
        scale_unit,
        magnification_text,
        *zoom_arrows,
        title_text,
    ]

# ==========================================
# 3. RENDER & SAVE TO MP4
# ==========================================
if __name__ == "__main__":
    print("Rendering Phase 1: Detector Geometry...")
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=ANIMATION_FRAMES,
        init_func=init,
        interval=1000 / FPS,
        blit=True,
    )
    
    output_filename = "smt_1.mp4"
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Max Lian'), bitrate=1800)
    
    anim.save(output_filename, writer=writer)
    plt.close()
    print(f"Done! Saved: {output_filename}")