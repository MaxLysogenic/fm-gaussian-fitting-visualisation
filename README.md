# Single-Molecule Localization Microscopy (SMLM) Simulation Pipeline

A 5-phase forward physical simulation and visualization pipeline illustrating Single-Molecule Localization Microscopy (SMLM) data processing, built using Python (`matplotlib`, `numpy`) and `ffmpeg`.

Developed for an academic presentation on single-molecule transcription dynamics in live cells.

### 🔬 Simulation Arc

1. **Phase 1 (`smt_1.py`): Pixel Size Scaling**  
   Magnifies a physical $16\,\mu\text{m}$ sCMOS sensor pixel by $150\times$ down to an effective sub-diffraction spatial scale of **$107\text{ nm}$ per pixel**.
2. **Phase 2 (`smt_2.py`): Photon Distribution Simulation**  
   Samples 2,000 in-grid signal photons from the optical point-spread function ($\text{FWHM} \approx 200\text{ nm}$) and 350 uniform background photons, animates their arrival, and saves the reproducible sample to `simulated_photons.npz`.
3. **Phase 3 (`smt_3.py`): Camera Digitization**  
   Cross-fades those arrivals into the $6 \times 6$ binned camera grid, adds reproducible Gaussian read noise, and displays the rounded pixel counts.
4. **Phase 4 (`smt_4.py`): Gaussian Surface Fitting**  
   Fits a continuous 2D Gaussian to the digitized pixel counts using non-linear least squares and overlays fitted contours.
5. **Phase 5 (`smt_5.py`): Sub-Pixel Localization**  
   Reports the fitted centroid and shows the **$\pm 30\text{ nm}$ localization uncertainty** circle ($1\sigma$).

From the repository root, render the phases in order with `python gaussian_fitting_visualisation/smt_1.py` through `python gaussian_fitting_visualisation/smt_5.py`. Videos are saved in `gaussian_fitting_visualisation/generated_videos/`. To combine them after rendering all five phases, run `python gaussian_fitting_visualisation/file_stitch_videos.py`; the combined video is saved in gaussian_fitting_visualisation/generated_videos/. Phases 3–5 load the photon sample written by Phase 2, so rerun Phase 2 whenever you want to generate a new sample.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* `ffmpeg` installed and available in your system `PATH` (required for video rendering and concatenation).

### Setup
```bash
pip install -r requirements.txt  
```

## 📚 Acknowledgments & Physical Parameters
Camera hardware parameters ($16\,\mu\text{m}$ sensor pixel size, $150\times$ magnification) inspired by
Nguyen et al., 2021.

## 📜 License
MIT License

## References 
Nguyen, V. Q., Anand, R., Sheng, L., Tang, X., Yick Hin, L., Wisniewski, J., Mizuguchi, G., Li, K. Y., Jou, V., Zheng, Q., Lavis, L. D., Lionnet, T., & Wu, C. (2021). Spatiotemporal coordination of transcription preinitiation complex assembly in live cells. Molecular Cell, 81(17), 3560–3575.e6. https://doi.org/10.1016/j.molcel.2021.07.022
