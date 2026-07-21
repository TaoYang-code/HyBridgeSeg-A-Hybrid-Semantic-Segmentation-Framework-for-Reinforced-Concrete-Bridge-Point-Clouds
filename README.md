# Bridge Segmentation Release

This repository contains a cleaned release version of the bridge point-cloud segmentation pipeline.

The project is organized around one main entrypoint:

```bash
python run_step1_to_final.py
```

The pipeline processes raw bridge point clouds one bridge at a time and produces the final component-level segmentation results.

## Project Structure

- `run_step1_to_final.py`
  Full end-to-end pipeline runner.
- `src/`
  Core implementation of the five pipeline stages.
- `vendor/`
  Trimmed third-party/reference code required by the released pipeline.
- `data_input/`
  Input bridge point clouds in `.txt` format.
- `checkpoints/`
  Required pretrained model weights.

Runtime outputs are written under `runs/` when the pipeline is executed.

## Pipeline Overview

The full workflow includes:

1. Raw point-cloud loading
2. Longitudinal splitting into five bridge sections
3. Module 1 pre-segmentation into `background`, `sub`, `deck`, and `sup_withoutdeck`
4. Refinement of `sub`, `deck`, and `sup_withoutdeck`
5. Superstructure processing
6. Substructure processing with classifier-assisted pier extraction
7. Final result integration

The superstructure and substructure branches run in parallel by default.

## Input and Checkpoints

Default paths:

- Input point clouds: `data_input/`
- Module 1 checkpoint: `checkpoints/Pre-segmentation_in_Module-1.pth`
- Module 2 checkpoint: `checkpoints/classification_in_Module-2.pth`

## Usage

Run the complete pipeline:

```bash
python run_step1_to_final.py
```

Save intermediate files:

```bash
python run_step1_to_final.py --save-intermediate
```

Run the two main branches sequentially instead of in parallel:

```bash
python run_step1_to_final.py --sequential-branches
```

Specify a different Python executable for vendor-model inference:

```bash
python run_step1_to_final.py --python-exe "C:\\path\\to\\python.exe"
```

## Main Outputs

After execution, the final segmentation results are stored in:

```text
runs/step5_final_output/whole_result/
```

Per-bridge timing statistics are saved to:

```text
runs/step5_final_output/pipeline_timing_summary.json
```

## Environment Notes

Before running the project, make sure the environment provides the dependencies required by the pipeline and the bundled vendor code, including PyTorch, Open3D, NumPy, and the point-cloud extensions required by the referenced models.
