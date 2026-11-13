<img src="resources/images/logo.png" alt="CADS-logo" width="30%">

# CADS : A Comprehensive Anatomical Dataset and Segmentation for Whole-Body Anatomy in Computed Tomography

## News
- [October 2026] 🎉 CADS is now published in **Nature Communications**! A happy chapter in the CADS story, with more to come! 🌱
- [July 2026] New model variants are now available! We've introduced licensing-stratified options and the anatomy-driven CADS-region model suite to provide more flexibility for different use cases. For more details, please see the [Model Variants](#model-variants) section.
- [July 2025] Updated model weights and paper preprint, please refer to the [Useful Links](#useful-links) section below!
- [March 2025] Model weights and paper preprint coming soon!

## Table of Contents
1. [Overview](#overview)
2. [Useful Links](#useful-links)
3. [Installation](#installation)
4. [Model Variants](#model-variants)
5. [Option 1: Quick Start](#option-1--quick-start-run-everything-in-one-command)
6. [Option 2: Staged Pipeline](#option-2--staged-pipeline-recommended-for-large-scale-processing)
7. [Target Structures in Each Task](#target-structures-in-each-task)
8. [Other Resources for Download](#other-resources-for-download)
9. [License](#license)
10. [Citation](#citation)

## Overview
<img src="resources/images/whole-body-parts-visualization.svg" alt="overview" width="100%">

CADS is a robust, fully automated framework for segmenting 167 anatomical structures in Computed Tomography (CT), spanning from head to knee regions across diverse anatomical systems.

The framework consists of two main components:

1. **CADS-dataset**:
   - 22,022 CT volumes with complete annotations for 167 anatomical structures.
   - Most extensive whole-body CT dataset, exceeding current collections in both scale (18x more CT scans) and anatomical coverage (60% more distinct targets).
   - Data collected from publicly available datasets and private hospital data, spanning 100+ imaging centers across 16 countries.
   - Diverse coverage of clinical variability, protocols, and pathological conditions.
   - Built through an automated pipeline with pseudo-labeling and unsupervised quality control.

2. **CADS-model**:
   - An open-source model suite for automated whole-body segmentation.
   - Performance validated on both public challenges and real-world hospital cohorts.
   - Available as Python script run (this GitHub repo) for flexible command-line usage.
   - Also available as a user-friendly 3D Slicer plugin with UI interface, simple installation and one-click inference.

## Useful Links
- [📄 CADS Paper](https://doi.org/10.1038/s41467-026-78732-1)
- [🤗 CADS-dataset](https://huggingface.co/datasets/mrmrx/CADS-dataset)
- [📦 CADS-model Weights](https://github.com/murong-xu/CADS/releases)
- [🔧 CADS-model Codebase](https://github.com/murong-xu/CADS)
- [🛠 CADS-model 3D Slicer Plugin](https://github.com/murong-xu/SlicerCADSWholeBodyCTSeg)

## Installation

Below we provide instructions for Python script run. For 3D Slicer plugin installation, please refer to [SlicerCADSWholeBodyCTSeg](https://github.com/murong-xu/SlicerCADSWholeBodyCTSeg).

```bash
# 1. Clone the repository
git clone git@github.com:murong-xu/CADS.git
# or download from https://github.com/murong-xu/CADS

# 2. Create and activate conda environment (Python>=3.9 required)
conda create -n CADS_env python=3.11
conda activate CADS_env

# 3. Install PyTorch
# Visit https://pytorch.org/ and select the appropriate version based on your Operating System and CUDA version
# Browse previous releases here if you prefer not to install the latest version: https://pytorch.org/get-started/previous-versions/
# CADS requirement: torch>=2.1.2
# Example installation command for Linux with CUDA 12.4:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install CADS
cd CADS  # the cloned git repo
pip install -e .

# Note: During installation, you may see warnings about "DEPRECATION: Legacy editable install".
# This warning is expected and can be safely ignored - it doesn't affect the functionality of CADS.
```

## Model Variants

CADS provides three license-stratified pretrained model-weight variants. They use the same architecture, task grouping, and 167 target structures, but differ in the licensed training data sources.

- **`reference`**: CADS customized model license. Intended for reproducing the reference CADS-model and manuscript results, or for users who specifically need the model trained on the full CADS data-source collection.
- **`research`**: CC BY-NC-SA 4.0. Recommended for most academic and non-commercial research use, with performance nearly identical to the reference model in the reported evaluation.
- **`open`**: CC BY-SA 4.0. Intended for users who need a model-weight license permitting redistribution of derived weights, including commercial use, under share-alike terms. Performance may be degraded for some detailed head-and-neck structures.

CADS also provides **`CADS-region`**, an anatomy-driven model suite that reorganizes the same 167 structures into region-specific tasks for practical deployment by anatomical field of view or downstream analysis. CADS-region is distributed under the same CADS customized model license as the reference model and is available from the [CADS-region release](https://github.com/murong-xu/CADS/releases/tag/cads-model-region_v1.0.0). For the complete label map and target list for each task, please refer to [cads-region_labelmap](https://github.com/murong-xu/CADS/blob/main/resources/info/cads-region_labelmap.md).

For details about the licensing terms and model differences, see [cads/LICENSE_MODEL](cads/LICENSE_MODEL) and the [CADS model releases](https://github.com/murong-xu/CADS/releases).


## Option 1 — Quick Start (Run everything in one command)
Example script for running inference:

```bash
python cads/scripts/predict_images.py \
    -in "/path/to/ct_images" \
    -out "/path/to/output" \
    -task all \
    -license research
```

`-task` can be `all` or a single task ID from 551-559. 

`-license` can be `reference`, `research`, or `open`; if omitted, the default is `reference`. 

Add `--cpu` to run without GPU acceleration.


## Option 2 — Staged Pipeline (Recommended for large-scale processing)
For large-scale processing of CT volumes (e.g., ~100–1000 scans or more), which can require more processing time, the full pipeline can be split into **standalone** stages to better utilize computational resources: **CPU** for CT preprocessing and segmentation restoration, and **GPU** for model inference. Intermediate outputs are saved to disk for use in subsequent stages.
1. **Preprocess CT (CPU-only)** — `cads/scripts/run_01_preprocess.py`
   Reads raw CT NIfTI files, performs reorientation and resampling, and writes preprocessed images along with a small *_metadata.pkl file per case (required later for restoration).
   ```bash
   python cads/scripts/run_01_preprocess.py \
    -in "/path/to/raw_ct_images" \  # Directory of raw CT nifti images
    -out_preprocessed_img "/path/to/preprocessed" \  # Output directory for preprocessed images
    -out_metadata "/path/to/metadata" \  # Output directory for image metadata
    ```
2. **Inference (GPU recommended)** — `cads/scripts/run_02_inference.py`
   Runs the CADS-model on **preprocessed** NIfTIs only. Set `-task` to `all` or a single id from 551–559. Outputs one folder per case with `_part_551`…`_part_559` segmentations in **preprocessed** space. Add `--cpu` if no GPU is available.
   ```bash
   python cads/scripts/run_02_inference.py \
    -in_preprocessed_images "/path/to/preprocessed" \
    -out "/path/to/seg_in_preprocessed_space" \
    -task all \
    -license research
   ```
   `-license` is optional and defaults to `reference`. 
   
   Add `--cpu` to run without GPU acceleration.
3. **Restore segmentation to original image format (CPU-only)** — `cads/scripts/run_03_restore_seg.py`
   Uses the inference outputs (from step 2) and image metadata (from step 1) to restore segmentations to the **original** image geometry.
   ```bash
   python cads/scripts/run_03_restore_seg.py \
    -in_seg "/path/to/seg_in_preprocessed_space" \
    -in_metadata "/path/to/metadata" \
    -out_seg "/path/to/seg_in_original_space" \  # Output segmentation masks in original image geometry
   ```

## Device Selection
All three inference scripts (`predict_images.py`, `run_02_inference.py`, `predict_slicer.py`) accept `-device`:

- `auto` (default): CUDA if available, then Intel XPU, then CPU. If `CADS_DEVICE` is set (e.g. `CADS_DEVICE=xpu`), it is used instead.
- `cuda`, `xpu` or `cpu`, optionally with an index (e.g. `cuda:1`). A device that is not available falls back to CPU.
- `openvino` or `openvino:<OV device>`: OpenVINO backend, see below.
- `--cpu` still forces CPU.

With `auto` mode, models are preloaded on the GPU only if the free device memory can be queried and is large enough; otherwise weights stay on CPU and are moved to the GPU per task.

Intel XPU requires a PyTorch build with XPU support, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

To compare two segmentations of the same case and task (e.g. a CPU run and a GPU run) as Dice per structure:
```bash
python cads/scripts/compare_segs.py -a cpu/case/case_part_558.nii.gz -b xpu/case/case_part_558.nii.gz -task 558 -out dice_558.csv
```

### OpenVINO backend (optional)
`-device openvino` runs the network forward pass with the [OpenVINO](https://github.com/openvinotoolkit/openvino) CPU plugin; `-device openvino:<OV device>` (e.g. `openvino:GPU`) selects another OpenVINO device, and `CADS_DEVICE=openvino` works as well. Only the forward pass is swapped: nnU-Net 2.5.2 (unmodified) still does the preprocessing, sliding window, Gaussian weighting, mirroring (if enabled) and export, and CADS does its own pre- and post-processing.

```bash
pip install -e ".[openvino]"   # or: pip install "openvino>=2025.3"
# optional: cache the IR locally (otherwise each run converts the network in memory at start-up)
python cads/scripts/export_openvino.py -license open -task 552 553 556 558          # FP32 IR
python cads/scripts/export_openvino.py -license open -task 552 553 556 558 -fp16    # FP16-compressed IR
python cads/scripts/predict_images.py -in ct.nii.gz -out out -task 558 -license open -device openvino
CADS_OV_IR=fp16 python cads/scripts/predict_images.py -in ct.nii.gz -out out_fp16 -task 558 -license open -device openvino
```

- The IR is written next to each checkpoint (`<weights>/<variant>/Dataset*/<trainer>/fold_all/openvino/checkpoint_final[_fp16].xml/.bin`), with a fixed input of one patch (`1 x C x patch size`), which is what the sliding window feeds. It is derived from the model weights and carries their licence: for `-license open` that is the CADS open weights (`cads-model-open_v1.0.0`, CC BY-SA 4.0, by the CADS authors). Export it locally from the downloaded weights; do not commit it to this repository.
- On CPU the OpenVINO inference precision is pinned to FP32 (`CADS_OV_PRECISION_HINT=f32`), so CPUs with bf16 support do not switch precision on their own. `CADS_OV_PRECISION_HINT=default` keeps the plugin default. The FP16 IR only compresses the stored weights; on CPU they are still computed in FP32.
- `CADS_LOG_FORWARD_TIME=1` prints the time spent in the network forward pass per model for the PyTorch devices; the OpenVINO backend always prints it.

## Target Structures in Each Task
Each task ID (model 551-559) represents a specific anatomical group. For detailed indexing please refer to [model labelmap](resources/info/labelmap.md).

**Task 551: Major abdominal organs, primary thoracic organs (lungs), and major abdominal vasculature**
Spleen, Kidney R/L, Gallbladder, Liver, Stomach, Aorta, Inferior vena cava, Portal and splenic vein, Pancreas, Adrenal gland R/L, Upper/Lower lobe of left lung, Upper/Middle/Lower lobe of right lung.

**Task 552: Complete set of individual vertebrae from cervical to lumbar regions**
Vertebrae C1-C7, T1-T12, L1-L5.

**Task 553: Various thoracic and abdominal organs, brain, major pelvic vessels, and face**
Esophagus, Trachea, Myocardium, Heart atrium R/L, Heart ventricle R/L, Pulmonary artery, Brain, Common iliac artery R/L, Common iliac vein R/L, Small intestine, Duodenum, Colon, Urinary bladder, Face.

**Task 554: Major bones of the appendicular skeleton, sacrum, and associated large muscle groups**
Humerus R/L, Scapula R/L, Clavicle R/L, Femur R/L, Hip stucture R/L, Bone structure of sacrum, Gluteus maximus/medius/minimus muscles R/L, Deep muscle of back R/L, Iliopsoas muscle R/L.

**Task 555: Complete set of individual ribs, both left and right**
Ribs 1-12 R/L.

**Task 556: Miscellaneous structures for radiation therapy**
Spinal canal, Larynx, Heart, Bowel space, Sigmoid colon, Rectum, Prostate, Seminal vesicle, Mammary gland R/L, Sternum, Psoas major muscle R/L, Rectus abdominis muscle R/L.

**Task 557: Brain and head tissues**
White matter, Gray matter, Cerebrospinal fluid, Scalp, Eyeballs, Compact bone, Spongy bone, Vascular structure of head, Rectus eye muscle structure.

**Task 558: Head and neck structures**
Carotid artery R/L, Arytenoid cartilage, Mandible, Brain stem, Buccal mucosa, Oral cavity, Cochlea R/L, Cricopharyngeus, Cervical esophagus, Anterior/Posterior eyeball segment R/L, Lacrimal gland R/L, Submandibular gland R/L, Thyroid, Glottis, Supraglottis, Lip, Optic chiasm, Optic nerve R/L, Parotid gland R/L, Pituitary gland.

**Task 559: General tissue types, major body cavities, broad anatomical categories**
Subcutaneous tissue, Muscle, Abdominopelvic cavity, Thoracic cavity, Bone structure, Gland structure, Pericardium, Breast implant, Mediastinum, Spinal cord.

## Other Resources for Download

Additional resources released with the CADS manuscript are available here:

- **Expert Review and Corrected Label Subset on TotalSegmentator Data**: Radiologist review records for the 65 held-out TotalSegmentator test cases used in the CADS study, together with manually corrected labels used for the corrected mini-benchmark. [[Download]](https://drive.switch.ch/index.php/s/pc9mJaYLzoYSfwe)

## Measurements

Tested on: ARL Core Ultra 9 285H / Arc 140T (CPU and XPU), 64 GB RAM, Windows 11 (build 26200), Intel graphics driver 32.0.101.9033, Python 3.11.18, torch 2.7.1+xpu.

Test CT: TCGA-THCA, patient TCGA-DE-A4MA, series "CT HeadNeck 3.0 B31s" (SeriesInstanceUID `1.3.6.1.4.1.14519.5.2.1.8421.4019.123670597687203511219463516732`), CC BY 3.0, from [TCIA](https://doi.org/10.7937/K9/TCIA.2016.9ZFRVF1B) via [IDC](https://portal.imaging.datacommons.cancer.gov/). Weights: `-license open` (CC BY-SA 4.0). Task 558 also runs tasks 552 and 553, which its post-processing uses.

Setup (Windows, PowerShell, from the repository root):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/xpu
pip install -e . idc-index "pyarrow<20"
mkdir ct, logs -Force
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '-')"
idc download-from-selection --series-instance-uid 1.3.6.1.4.1.14519.5.2.1.8421.4019.123670597687203511219463516732 --download-dir ct_dicom --dir-template ""
python -c "import SimpleITK as s, glob; r=s.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames('ct_dicom')); s.WriteImage(r.Execute(), 'ct/tcga_de_a4ma_0000.nii.gz')"
```

`measure.bat` (runs CPU and XPU for tasks 556 and 558, then the Dice comparison; each XPU run is repeated, the first as a warm-up):
```bat
cd /d %~dp0
call .venv\Scripts\activate.bat
set W=-in ct\tcga_de_a4ma_0000.nii.gz -license open
for %%T in (556 558) do (
  python cads\scripts\predict_images.py %W% -out out\cpu_%%T -task %%T -device cpu > logs\cpu_%%T.log 2>&1
  python cads\scripts\predict_images.py %W% -out out\xpu_%%T_warm -task %%T -device xpu > logs\xpu_%%T_warm.log 2>&1
  python cads\scripts\predict_images.py %W% -out out\xpu_%%T -task %%T -device xpu > logs\xpu_%%T.log 2>&1
  python cads\scripts\compare_segs.py -a out\cpu_%%T\tcga_de_a4ma\tcga_de_a4ma_part_%%T.nii.gz -b out\xpu_%%T\tcga_de_a4ma\tcga_de_a4ma_part_%%T.nii.gz -task %%T -out logs\dice_%%T.csv > logs\dice_%%T.log 2>&1
)
```

Run it once as a scheduled task in the desktop session, then read `Finished in` (per case) and `_nnUNetv2_predict took` (per model) from `logs\*.log`:
```powershell
schtasks /create /tn cads_measure /tr "$PWD\measure.bat" /sc once /st 23:59 /it /rl HIGHEST /f
schtasks /run /tn cads_measure
schtasks /delete /tn cads_measure /f   # after it has finished
```

Results on ARL Core Ultra 9 285H / Arc 140T. With `-device xpu`, auto mode found 33.6 GB of free device memory (`torch.xpu.mem_get_info`) and chose `preload-gpu`. Wall time is the whole process (Python start, model loading, pre- and post-processing). `Finished in` and model inference are taken from the log. Peak memory is the working set of the process tree, sampled every 0.5 s.

| Task | Device | Run | Wall time (s) | `Finished in` (s) | Model inference (s) | Peak memory (GB) |
|---|---|---|---|---|---|---|
| 556 | CPU | | 128.9 | 97.9 | 86.5 | 10.6 |
| 556 | XPU | first | 58.9 | 38.6 | 26.7 | 12.1 |
| 556 | XPU | second | 48.2 | 37.4 | 26.5 | 12.0 |
| 558 (+552, 553) | CPU | | 343.9 | 329.2 | 86.8 + 85.7 + 141.4 | 12.7 |
| 558 (+552, 553) | XPU | first | 126.4 | 112.5 | 27.4 + 25.9 + 43.4 | 17.3 |
| 558 (+552, 553) | XPU | second | 126.3 | 112.4 | 27.5 + 25.9 + 43.4 | 18.0 |

CPU vs XPU (second run), Dice per structure present in either segmentation:

| Part | Structures present | Dice min | Dice mean |
|---|---|---|---|
| 556 | 3 / 15 | 1.0000 | 1.0000 |
| 558 | 23 / 29 | 1.0000 | 1.0000 |
| 552 | 12 / 24 | 1.0000 | 1.0000 |
| 553 | 4 / 18 | 0.999997 (brain: 300,637 vs 300,639 voxels) | 0.999999 |

The 556 and 558 label maps from CPU and XPU are byte-identical.

### OpenVINO backend on CPU

Tested on: Intel Xeon (Linux VM, 8 vCPU, 16 GB RAM, AVX-512/AMX), Python 3.11.17, torch 2.5.1+cpu, OpenVINO 2026.4.1, CPU plugin with FP32 inference precision. Same test CT and `-license open` weights as above. The VM is shared, so the times varied noticeably between repeated runs; both runs are listed. "Network forward" is the time spent in the network forward pass (`CADS_LOG_FORWARD_TIME=1` for PyTorch, always printed for OpenVINO); peak memory is the RSS of the process tree, sampled every 0.5 s.

| Task | Backend | Run | `Finished in` (s) | Model inference (s) | Network forward (s) | Peak memory (GB) |
|---|---|---|---|---|---|---|
| 556 | PyTorch CPU | 1 | 102.7 | 87.6 | 81.5 | 9.3 |
| 556 | PyTorch CPU | 2 | 100.0 | 85.2 | 78.0 | 9.4 |
| 556 | OpenVINO CPU, FP32 IR | 1 | 88.5 | 73.6 | 66.5 | 8.3 |
| 556 | OpenVINO CPU, FP32 IR | 2 | 61.4 | 47.5 | 41.2 | 9.0 |
| 556 | OpenVINO CPU, FP16 IR | 1 | 65.6 | 50.7 | 44.7 | 9.1 |
| 558 (+552, 553) | PyTorch CPU | 1 | 438.1 | 159.4 + 75.7 + 183.0 | 152.4 + 70.8 + 134.4 | 11.6 |
| 558 (+552, 553) | PyTorch CPU | 2 | 627.5 | 77.8 + 83.8 + 445.1 | 71.2 + 78.5 + 164.4 | 11.4 |
| 558 (+552, 553) | OpenVINO CPU, FP32 IR | 1 | 408.3 | 58.8 + 199.0 + 127.3 | 51.9 + 193.7 + 89.2 | 12.2 |
| 558 (+552, 553) | OpenVINO CPU, FP32 IR | 2 | 512.1 | 55.4 + 50.8 + 382.8 | 48.4 + 45.5 + 94.5 | 12.0 |
| 558 (+552, 553) | OpenVINO CPU, FP16 IR | 1 | 210.9 | 51.8 + 41.5 + 97.5 | 45.4 + 36.5 + 86.5 | 12.3 |

Model inference and network forward are listed per model in the order 552 + 553 + 558. Repeated runs of the same backend gave byte-identical label maps. The IR export (`export_openvino.py`, all four tasks) took 44 s (FP32) and 41 s (FP16) with a peak memory of 3.8 GB.

PyTorch CPU vs OpenVINO CPU, Dice per structure present in either segmentation (no structure below 0.99):

| Part | Structures present | FP32 IR: Dice min / mean (differing voxels) | FP16 IR: Dice min / mean (differing voxels) |
|---|---|---|---|
| 556 | 3 / 15 | 0.9949 / 0.9982 (16) | 0.9990 / 0.9996 (10) |
| 558 | 23 / 29 | 0.9996 / 0.9999 (51) | 0.9984 / 0.9998 (63) |
| 552 | 12 / 24 | 0.9993 / 0.9998 (42) | 0.9993 / 0.9998 (42) |
| 553 | 4 / 18 | 0.9923 / 0.9980 (139) | 0.9911 / 0.9977 (151) |

The lowest values are on small or thin structures: left mammary gland in 556 (492 voxels; 497 with FP32 IR, 493 with FP16 IR) and face in 553 (2,616 voxels; 2,576 with FP32 IR, 2,570 with FP16 IR). Differing voxels are counted over the whole 512 x 512 x 148 label map.

Open weights on head-and-neck structures (Task 558): per the CADS supplementary source data (Supplementary Fig. 19), six targets are unsupported in `open` because their only ground truth comes from HaN-Seg (CC BY-NC-ND 4.0): optic nerve L, optic chiasm, anterior eyeball segment L, cochlea L, lacrimal gland L and R. Their label IDs remain in the Task 558 label map; on the test CT above, all six were empty with the `open` weights.

## License

- **Codebase** (the `cads` package and all source code in this repository) is licensed under the [Apache License 2.0](LICENSE).
- **Model weights** include three license-stratified variants: `reference` under the CADS customized model license, `research` under CC BY-NC-SA 4.0, and `open` under CC BY-SA 4.0. CADS-region is released separately under the same CADS customized model license as `reference`. See [cads/LICENSE_MODEL](cads/LICENSE_MODEL) for details.

## Citation
If you use any component of CADS (CADS-dataset, its curated segmentation masks, pretrained CADS-model, or the 3D Slicer extension), please cite:

```bibtex
@article{xu2026cads,
  title={CADS: a comprehensive anatomical dataset and segmentation for whole-body anatomy in computed tomography},
  author={Xu, Murong and Navarro, Fernando and Fritsak, Maksym and Hamamci, Ibrahim Ethem and Shit, Suprosanna and Wittmann, Bastian and Er, Sezgin and Christ, Sebastian M. and de la Rosa, Ezequiel and Deseoe, Julian and Graf, Robert and Möller, Hendrik and Sekuboyina, Anjany and Peeken, Jan C. and Becker, Sven S. and Baldini, Giulia and Haubold, Johannes and Nensa, Felix and Hosch, René and Mirajkar, Nikhil and Khalid, Saad and Zachow, Stefan and Weber, Marc-André and Langs, Georg and Wasserthal, Jakob and Ozdemir, Mehmet Kemal and Fedorov, Andrey and Kikinis, Ron and Tanadini-Lang, Stephanie and Kirschke, Jan S. and Combs, Stephanie E. and Amiranashvili, Tamaz and Menze, Bjoern},
  journal={Nature Communications},
  year={2026},
  doi={10.1038/s41467-026-78732-1},
  url={https://doi.org/10.1038/s41467-026-78732-1}
}
```