# Tools

Python scripts need numpy, scipy, matplotlib, imageio and imageio-ffmpeg.

| Script | Purpose |
|---|---|
| `mallard_vtu.py` | Minimal reader for Mallard's VTU files (cells, cell data, time) |
| `plot_vtu.py` | Plot one cell field of a VTU file |
| `compare_png.py` | Side-by-side plots of a field from several VTU files |
| `plot_sod.py` | Density profiles against the exact Sod solution |
| `animate.py` | MP4/GIF animation of a VTU series: density (or `--var`, including `VORTICITY`) with contours, plus numerical schlieren |
| `make_cylinder_mesh.py` | Gmsh O-grid around a cylinder |
| `zero_priority.py` | Run a command on a shared A100 node at zero priority: refuses non-A100 GPUs, waits for an idle node, and kills the job as soon as anyone else uses, reserves or queues for a GPU (`test_zero_priority.py` tests it) |
| `gpu_run.sh` | Sync, build for CUDA (Ampere) and run a case on an A100 node through `zero_priority.py` |
