import os
import sys
import torch
import time
import traceback
from pathlib import Path
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
import gc
from contextlib import contextmanager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from cads.dataset_utils.bodyparts_labelmaps import labelmap_all_structure, map_taskid_to_labelmaps, except_labels_combine
from cads.dataset_utils.preprocessing import preprocess_nifti, restore_seg_in_orig_format
from cads.dataset_utils.postprocessing import _do_outlier_postprocessing_groups, postprocess_head, postprocess_head_and_neck
from cads.utils.snapshot import generate_snapshot
from cads.utils.libs import time_it, cleanup_temp_files
from cads.dataset_utils.TPTBox import postprocess_seg_TPTBox
from cads.utils.openvino_backend import parse_openvino_device

TRAINERS = {
    551: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    552: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    553: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    554: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    555: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    556: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    557: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    558: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
    559: 'nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres',
}

def get_task_model_folder(task_id: int, model_folder: str):
    """
    Find the sub-directory containing 'nnUNetResEncUNetLPlans' for a given task in the model folder.
    """
    model_folder_path = Path(model_folder)
    dirs = [d for d in model_folder_path.glob("*") if d.is_dir()]
    for d in dirs:
        if f'Dataset{task_id:03d}' in d.name:
            subfolder = d.resolve()
            subdirs = [sd for sd in subfolder.glob(
                "*") if sd.is_dir() and 'nnUNetResEncUNetLPlans' in sd.name]
            trainer = TRAINERS[task_id]
            if subdirs:
                for subdir in subdirs:
                    if trainer in subdir.name:
                        print(f'Inference using model {subdir.name}')
                        return str(subdir.resolve())
                else:
                    raise ValueError(
                        f"No subdirectory containing {trainer} found in {subfolder}")
            else:
                raise ValueError(
                    f"No subdirectory containing 'nnUNetResEncUNetLPlans' found in {subfolder}")

    raise ValueError(
        f'task id {task_id} not found in model folder {model_folder}')


class nnUNetv2Predictor():
    """
    A class to handle predictions using nnUNetv2 models.
    This class initializes an nnUNetv2 predictor for a specified task and provides methods for both batch and single image predictions.
    """

    def __init__(self, model_folder, task_id, device, batch_predict=False, folds='all', checkpoint='checkpoint_final.pth', num_threads_preprocessing=6, num_threads_nifti_save=2, verbose=False, ov_device=None):
        self.num_threads_preprocessing = num_threads_preprocessing
        self.num_threads_nifti_save = num_threads_nifti_save
        self.device = device

        # Use default settings recommended by nnUNet (also same as TotalSeg_v2)
        self.predictor = nnUNetPredictor(tile_step_size=0.5,  # TODO: 0.5, 0.8
                                         use_gaussian=True,
                                         use_mirroring=False,
                                         device=self.device,
                                         verbose=verbose,
                                         allow_tqdm=True,
                                         verbose_preprocessing=verbose)
        task_id = int(task_id)
        try:
            task_model_folder = get_task_model_folder(task_id, model_folder)
        except ValueError as e:
            print(e)
            sys.exit(1)

        self.predictor.initialize_from_trained_model_folder(
            task_model_folder, folds, checkpoint)

        # Optional OpenVINO backend: only the network forward is swapped, nnU-Net does the rest
        if ov_device is not None:
            from cads.utils.openvino_backend import attach_openvino
            attach_openvino(self.predictor, task_model_folder, folds, checkpoint, ov_device)

        # Optional timing of the network forward passes only (CADS_LOG_FORWARD_TIME=1)
        self._forward_timer = None
        if ov_device is None and os.environ.get('CADS_LOG_FORWARD_TIME', '').lower() in ('1', 'true'):
            self._forward_timer = _ForwardTimer(self.predictor.network)

        if batch_predict:
            self.predict = self._nnUNetv2_batch_predict
        else:
            self.predict = self._nnUNetv2_predict

    @time_it
    def _nnUNetv2_batch_predict(self, folder_in, folder_out):
        """
        Identical to command nnUNetv2_predict, works for batch predictions (predicting many images at once), supposed to be faster.
        """
        # This is suitable for processing a bunch of files, and will produce some side-files (predict_from_raw_data_args.json ...)
        self.predictor.predict_from_files(folder_in, folder_out, save_probabilities=False,
                                          overwrite=True,
                                          num_processes_preprocessing=self.num_threads_preprocessing,
                                          num_processes_segmentation_export=self.num_threads_nifti_save,
                                          folder_with_segs_from_prev_stage=None,
                                          num_parts=1, part_id=0)

    @time_it
    def _nnUNetv2_predict(self, file_in, file_out):
        """
        Identical to command nnUNetv2_predict, works for predicting one image at a time.
        """
        img, props = SimpleITKIO().read_images([file_in])
        self.predictor.predict_single_npy_array(
            img, props, None, file_out, False)
        self._report_forward_time()

    def _report_forward_time(self):
        timer = self._forward_timer if self._forward_timer is not None else self.predictor.network
        if hasattr(timer, 'release'):
            timer.release()  # OpenVINO: free the compiled model and its buffers until the next prediction
        if hasattr(timer, 'forward_s') and timer.n_forward:
            backend = 'openvino' if self._forward_timer is None else 'torch'
            print(f"[{backend}] network forward took {timer.forward_s:.2f} seconds for {timer.n_forward} patches")
            timer.reset_timer()


class _ForwardTimer:
    """Accumulates the wall time of a torch network's forward calls via hooks."""

    def __init__(self, network):
        self.forward_s, self.n_forward, self._t0 = 0.0, 0, None
        network.register_forward_pre_hook(self._pre)
        network.register_forward_hook(self._post)

    def _pre(self, module, args):
        self._t0 = time.perf_counter()

    def _post(self, module, args, output):
        if output.device.type == 'cuda':
            torch.cuda.synchronize()
        elif output.device.type == 'xpu':
            torch.xpu.synchronize()
        self.forward_s += time.perf_counter() - self._t0
        self.n_forward += 1

    def reset_timer(self):
        self.forward_s, self.n_forward = 0.0, 0


def save_target_to_nifti(target, seg, output_folder, labelmap_inv, basename, original_affine):
    include_value = labelmap_inv[target]
    mask = np.isin(seg, include_value)
    binary_img = np.where(mask, 1, 0)

    output_path = os.path.join(output_folder, f'{basename}_{target}.nii.gz')
    nib.save(nib.Nifti1Image(binary_img.astype(
        np.uint8), original_affine), output_path)


def save_targets_to_nifti(save_separate_targets, output_targets_dir, affine, labelmap, seg, patient_id, nr_threads_saving):
    """
    Save a specific target segmentation to a binary nifti file.
    """
    targets = [structure for structure in save_separate_targets if structure in list(
        labelmap.values())]
    if targets:
        labelmap_inv = {v: k for k, v in labelmap.items()}
        _ = p_map(partial(save_target_to_nifti, seg=seg, output_folder=output_targets_dir, labelmap_inv=labelmap_inv,
                  basename=patient_id, original_affine=affine), targets, num_cpus=nr_threads_saving)

def _xpu_available():
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def get_device(device='auto', use_cpu=False):
    """
    Resolve the torch device used for inference.
    device: 'auto', 'cuda', 'xpu' or 'cpu', optionally with an index (e.g. 'cuda:1', 'xpu:0').
        If 'auto' (or None), the CADS_DEVICE environment variable is used when set.
        'auto' picks CUDA if available, then Intel XPU, then CPU.
        A requested device that is not available falls back to CPU.
    """
    if use_cpu:
        return torch.device('cpu')
    if device is None or device == 'auto':
        device = os.environ.get('CADS_DEVICE', 'auto')
    device = str(device).strip().lower()

    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if _xpu_available():
            return torch.device('xpu:0')
        return torch.device('cpu')

    dev = torch.device(device)
    if (dev.type == 'cuda' and not torch.cuda.is_available()) or (dev.type == 'xpu' and not _xpu_available()):
        print(f"[predict] Requested device '{device}' is not available, falling back to CPU")
        return torch.device('cpu')
    if dev.type in ('cuda', 'xpu') and dev.index is None:
        dev = torch.device(dev.type, 0)
    return dev


def _device_mem_gb(device: torch.device):
    """
    Return (free, total) device memory in GB, or (None, None) if it cannot be queried.
    """
    try:
        if device.type == 'cuda':
            free_b, total_b = torch.cuda.mem_get_info(device)
        elif device.type == 'xpu' and hasattr(torch.xpu, 'mem_get_info'):
            free_b, total_b = torch.xpu.mem_get_info(device)
        else:
            return None, None
    except Exception:
        return None, None
    return free_b / 1024**3, total_b / 1024**3


def _empty_device_cache(device: torch.device):
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device.type == 'xpu' and _xpu_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


def setup_inference_devices(mode='auto', use_cpu=False, n_tasks=None, device='auto'):
    """
    Set up devices for model initialization and inference based on available resources and mode.
    mode: Inference mode, one of ['auto', 'cpu-cache', 'preload-gpu'].
        'auto': Automatically choose mode based on available GPU memory and workload.
            If free GPU memory cannot be queried, 'cpu-cache' is used.
        'cpu-cache': Load model weights on CPU, move to GPU for inference.
        'preload-gpu': Load model weights on GPU, perform inference on GPU.
    device: 'auto', 'cuda', 'xpu', 'cpu' (see get_device), or 'openvino[:<OV device>]' for the OpenVINO backend.
    """
    cpu = torch.device('cpu')
    ov_device = None if use_cpu else parse_openvino_device(device)
    if ov_device is not None:
        print(f"[predict] OpenVINO backend on {ov_device} | mode=openvino")
        return 'openvino', cpu, cpu
    gpu = get_device(device, use_cpu=use_cpu)

    if gpu.type == 'cpu':
        chosen_mode = 'cpu-only'
    else:
        if mode not in ('auto', 'cpu-cache', 'preload-gpu'):
            mode = 'auto'
        if mode == 'auto':
            high_vram_thresh_gb = 28.0  # I think a 28GB GPU should be safe for batch inference
            single_model_thresh_gb = 12.0  # also 12GB should be safe for a single task inference

            free_gb, total_gb = _device_mem_gb(gpu)
            # free memory unknown
            if free_gb is None:
                chosen_mode = 'cpu-cache'
            # VRAM is sufficient
            elif free_gb >= high_vram_thresh_gb:
                chosen_mode = 'preload-gpu'
            # intermediate VRAM（>=12GB) but single task, can still use preload-gpu
            elif n_tasks == 1 and free_gb >= single_model_thresh_gb:
                chosen_mode = 'preload-gpu'
            else:
                # no enough VRAM
                chosen_mode = 'cpu-cache'
        else:
            chosen_mode = mode

    # Set devices based on chosen mode
    if chosen_mode == 'cpu-only':
        device_init = cpu
        device_run = cpu
    elif chosen_mode == 'cpu-cache':
        device_init = cpu
        device_run = gpu
    else:  # preload-gpu mode
        device_init = gpu
        device_run = gpu

    if chosen_mode == 'cpu-only':
        print(f"[predict] CPU-only mode")
    else:
        free_gb, total_gb = _device_mem_gb(gpu)
        if free_gb is None:
            print(f"[predict] {gpu} available. free memory unknown | mode={chosen_mode}")
        else:
            print(f"[predict] {gpu} available. free={free_gb:.1f} GB / total={total_gb:.1f} GB | mode={chosen_mode}")

    return chosen_mode, device_init, device_run

@contextmanager
def use_device(pred, target_device: torch.device, restore_to: torch.device = None, clear_cuda_cache: bool = True):
    """
    Temporarily switch the device of the predictor to the target_device, and restore it back to restore_to (default is to the original device) upon exit.
    - pred: a nnUNetPredictor
    - clear_cuda_cache: if switching from a GPU (CUDA or XPU) to CPU, clear the device cache to free up memory
    """
    old_device = getattr(pred, "device", torch.device("cpu"))
    try:
        if old_device != target_device:
            pred.predictor.network = pred.predictor.network.to(target_device)
            pred.predictor.device = target_device
            pred.device = target_device
        yield pred
    finally:
        dst = old_device if restore_to is None else restore_to
        if getattr(pred, "device", target_device) != dst:
            pred.predictor.network = pred.predictor.network.to(dst)
            pred.predictor.device = dst
            pred.device = dst
        # cpu-cache mode, low VRAM case
        if clear_cuda_cache and target_device.type in ("cuda", "xpu") and dst.type == "cpu":
            try:
                _empty_device_cache(target_device)
                gc.collect()
            except Exception:
                pass

def predict(files_in, folder_out, model_folder, task_ids, 
            folds='all', use_cpu=False, preprocess_cads=True, postprocess_cads=True, 
            save_all_combined_seg=False, snapshot=False, save_separate_targets=False, 
            num_threads_preprocessing=4, nr_threads_saving=6, 
            mode='auto', verbose=False, device='auto'):
    """
    Loop images and use nnUNetv2 models to predict. 
    """
    labelmap_all_structure_inv = {v: k for k,
                                  v in labelmap_all_structure.items()}

    # Ensure task_ids include dependencies
    task_ids = list(sorted(task_ids))
    if any(t in task_ids for t in (557, 558)) and 553 not in task_ids:
        task_ids.append(553)
    if 558 in task_ids and 552 not in task_ids:
        task_ids.append(552)
    task_ids.sort()

    # Setup device and inference mode
    chosen_mode, device_init, device_run = setup_inference_devices(mode=mode, use_cpu=use_cpu, n_tasks=len(task_ids), device=device)

    # Init nnUNetv2 predictor
    models = {}
    for task_id in task_ids:
        models[task_id] = nnUNetv2Predictor(model_folder, task_id, device_init, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth',
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose,
                                            ov_device=parse_openvino_device(device) if chosen_mode == 'openvino' else None)

    # Loop images
    for i, file_in in enumerate(files_in):
        if os.path.basename(file_in)[-11:] == "0000.nii.gz":
            patient_id = os.path.basename(file_in)[:-12]
        else:
            patient_id = os.path.basename(file_in)[:-7]
        print("Predicting file {}/{}   ".format(i+1, len(files_in)), patient_id)
        start = time.time()

        temp_dir = None
        if preprocess_cads:
            # Reorient to RAS, resampling to 1.5, remove rotation and translation
            temp_dir, file_in, metadata_orig, preprocessing_done = preprocess_nifti(file_in, spacing=1.5, num_threads_preprocessing=num_threads_preprocessing)

        output_dir = os.path.join(folder_out, patient_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Inference
        for task_id in task_ids:
            try:
                file_out = os.path.join(output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')
                with use_device(models[task_id], device_run, restore_to=device_init):
                    models[task_id].predict(file_in, file_out)
                    # models[task_id].predict([[file_in]], [file_out])  # for batch_predict

                if postprocess_cads:
                    try:
                        if task_id in _do_outlier_postprocessing_groups:
                            postprocess_seg_TPTBox(file_out, task_id, file_out)
                        if task_id in [557, 558]:
                            file_seg_brain_group = os.path.join(output_dir, patient_id+'_part_'+str(553)+'.nii.gz')
                            if not os.path.exists(file_seg_brain_group):
                                print(f"Task {task_id} needs pre-segmentation from task 553, generating segmentations...")
                                with use_device(models[553], device_run, restore_to=device_init):
                                    # models[553].predict([[file_in]], [file_seg_brain_group])  # for batch_predict
                                    models[553].predict(file_in, file_seg_brain_group)

                            # For group 558, also need cervical vertebrae as reference
                            if task_id == 558:
                                file_seg_vertebrae_group = os.path.join(output_dir, patient_id+'_part_'+str(552)+'.nii.gz')
                                # First check if brain exists and is sufficient
                                if not postprocess_head_and_neck(task_id, file_seg_brain_group, None, file_out):
                                    # Only predict vertebrae if brain check failed
                                    if not os.path.exists(file_seg_vertebrae_group):
                                        print(f"Task {task_id} needs pre-segmentation from task 552 (spine), generating segmentations...")
                                        with use_device(models[552], device_run, restore_to=device_init):
                                            # models[552].predict([[file_in]], [file_seg_vertebrae_group])  # for batch_predict
                                            models[552].predict(file_in, file_seg_vertebrae_group)
                                    postprocess_head_and_neck(task_id, file_seg_brain_group, file_seg_vertebrae_group, file_out)
                            else:
                                postprocess_head(task_id, file_seg_brain_group, file_out)
                    except Exception as e:
                        error_log = os.path.join(output_dir, f"{patient_id}_ERROR.log")
                        with open(error_log, 'a') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Postprocessing Error - Task {task_id}\n")
                            f.write(f"{'='*60}\n")
                            f.write(f"Error: {str(e)}\n\n")
                            f.write("Traceback:\n")
                            f.write(traceback.format_exc())
                            f.write("\n")

                        print(f"Warning: Postprocessing failed for task {task_id}: {e}")
                        traceback.print_exc()
            except Exception as e:
                error_log = os.path.join(output_dir, f"{patient_id}_ERROR.log")
                with open(error_log, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL ERROR - Task {task_id} FAILED\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Error: {str(e)}\n\n")                
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.write("    WARNING: Processing was interrupted. Remaining tasks and geometry restoration were not performed.\n")
                    f.write("    Existing output files are in preprocessed format (1.5mm spacing), NOT original image geometry.\n")
                    f.write("    Please fix the error and re-run to get outputs in original geometry.\n\n")

                print(f"Error: Task {task_id} failed: {e}")
                traceback.print_exc()
                print(f"Skipping task {task_id} and continuing with remaining tasks...")
                continue

        # reverse pre-processing
        for task_id in task_ids:
            file_out = os.path.join(output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')
            try:
                if preprocess_cads and preprocessing_done:
                    restore_seg_in_orig_format(file_out, file_out, metadata_orig, num_threads_preprocessing=num_threads_preprocessing)
            except Exception as e:
                error_log = os.path.join(output_dir, f"{patient_id}_ERROR.log")
                with open(error_log, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Restore Error - Task {task_id}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Error: {str(e)}\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.write(f"    WARNING: Failed to restore task {task_id} to original geometry.\n")
                    f.write(f"    Output file {file_out} remains in preprocessed format (1.5mm spacing).\n\n")

                print(f"  Error: Failed to restore task {task_id}: {e}")
                traceback.print_exc()

        # Combine all classes into a single segmentation nii file
        if save_all_combined_seg:
            init = False
            for task_id in task_ids:
                labelmap = map_taskid_to_labelmaps[task_id]
                seg = nib.load(os.path.join(
                    output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')).get_fdata()
                if not init:
                    seg_combined = np.zeros(seg.shape, dtype=np.uint8)
                    affine = nib.load(os.path.join(
                        output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')).affine
                    init = True
                for class_index, class_name in labelmap.items():
                    if class_name in except_labels_combine:
                        continue
                    seg_combined[seg ==
                                 class_index] = labelmap_all_structure_inv[class_name]

                # Save structures into separate binary segmentations
                if save_separate_targets:
                    output_targets_dir = os.path.join(output_dir, 'targets')
                    if not os.path.exists(output_targets_dir):
                        os.makedirs(output_targets_dir, exist_ok=True)
                    save_targets_to_nifti(save_separate_targets, output_targets_dir,
                                          affine, labelmap, seg, patient_id, nr_threads_saving)

            file_seg_combined = os.path.join(
                output_dir, patient_id+'_combined.nii.gz')
            nib.save(nib.Nifti1Image(seg_combined, affine), file_seg_combined)

        # Generate a snapshot of combined segmentations
        if snapshot:
            try:
                file_snapshot = os.path.join(
                    output_dir, patient_id + '_snapshot.png')
                if not os.path.isfile(file_seg_combined):
                    raise FileNotFoundError(
                        f"The file '{file_seg_combined}' does not exist.")
                generate_snapshot(file_in, file_seg_combined, file_snapshot)
            except FileNotFoundError as e:
                print(f"Error: {e}")

        if temp_dir:
            try:
                cleanup_temp_files(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp files in {temp_dir}: {e}")
        print(f"Finished in {time.time() - start:.2f}s")


def predict_preprocessed_images(files_in, folder_out, model_folder, task_ids, 
            folds='all', use_cpu=False, postprocess_cads=True, 
            num_threads_preprocessing=4, nr_threads_saving=6, 
            mode='auto', verbose=False, device='auto'):
    """
    Loop images and use nnUNetv2 models to predict. 
    """

    # Ensure task_ids include dependencies
    task_ids = list(sorted(task_ids))
    if any(t in task_ids for t in (557, 558)) and 553 not in task_ids:
        task_ids.append(553)
    if 558 in task_ids and 552 not in task_ids:
        task_ids.append(552)
    task_ids.sort()

    # Setup device and inference mode
    chosen_mode, device_init, device_run = setup_inference_devices(mode=mode, use_cpu=use_cpu, n_tasks=len(task_ids), device=device)

    # Init nnUNetv2 predictor
    models = {}
    for task_id in task_ids:
        models[task_id] = nnUNetv2Predictor(model_folder, task_id, device_init, batch_predict=False, folds=folds, checkpoint='checkpoint_final.pth',
                                            num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=nr_threads_saving, verbose=verbose,
                                            ov_device=parse_openvino_device(device) if chosen_mode == 'openvino' else None)

    # Loop images
    for i, file_in in enumerate(files_in):
        if os.path.basename(file_in)[-11:] == "0000.nii.gz":
            patient_id = os.path.basename(file_in)[:-12]
        else:
            patient_id = os.path.basename(file_in)[:-7]
        print("Predicting file {}/{}   ".format(i+1, len(files_in)), patient_id)
        start = time.time()

        output_dir = os.path.join(folder_out, patient_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Inference
        for task_id in task_ids:
            try:
                file_out = os.path.join(output_dir, patient_id+'_part_'+str(task_id)+'.nii.gz')
                with use_device(models[task_id], device_run, restore_to=device_init):
                    models[task_id].predict(file_in, file_out)
                    # models[task_id].predict([[file_in]], [file_out])  # for batch_predict

                if postprocess_cads:
                    try:
                        if task_id in _do_outlier_postprocessing_groups:
                            postprocess_seg_TPTBox(file_out, task_id, file_out)
                        if task_id in [557, 558]:
                            file_seg_brain_group = os.path.join(output_dir, patient_id+'_part_'+str(553)+'.nii.gz')
                            if not os.path.exists(file_seg_brain_group):
                                print(f"Task {task_id} needs pre-segmentation from task 553, generating segmentations...")
                                with use_device(models[553], device_run, restore_to=device_init):
                                    # models[553].predict([[file_in]], [file_seg_brain_group])  # for batch_predict
                                    models[553].predict(file_in, file_seg_brain_group)

                            # For group 558, also need cervical vertebrae as reference
                            if task_id == 558:
                                file_seg_vertebrae_group = os.path.join(output_dir, patient_id+'_part_'+str(552)+'.nii.gz')
                                # First check if brain exists and is sufficient
                                if not postprocess_head_and_neck(task_id, file_seg_brain_group, None, file_out):
                                    # Only predict vertebrae if brain check failed
                                    if not os.path.exists(file_seg_vertebrae_group):
                                        print(f"Task {task_id} needs pre-segmentation from task 552 (spine), generating segmentations...")
                                        with use_device(models[552], device_run, restore_to=device_init):
                                            # models[552].predict([[file_in]], [file_seg_vertebrae_group])  # for batch_predict
                                            models[552].predict(file_in, file_seg_vertebrae_group)
                                    postprocess_head_and_neck(task_id, file_seg_brain_group, file_seg_vertebrae_group, file_out)
                            else:
                                postprocess_head(task_id, file_seg_brain_group, file_out)
                    except Exception as e:
                        error_log = os.path.join(output_dir, f"{patient_id}_ERROR.log")
                        with open(error_log, 'a') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Postprocessing Error - Task {task_id}\n")
                            f.write(f"{'='*60}\n")
                            f.write(f"Error: {str(e)}\n\n")
                            f.write("Traceback:\n")
                            f.write(traceback.format_exc())
                            f.write("\n")

                        print(f"Warning: Postprocessing failed for task {task_id}: {e}")
                        traceback.print_exc()
            except Exception as e:
                error_log = os.path.join(output_dir, f"{patient_id}_ERROR.log")
                with open(error_log, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL ERROR - Task {task_id} FAILED\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Error: {str(e)}\n\n")                
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.write("    WARNING: Processing was interrupted. Remaining tasks and geometry restoration were not performed.\n")
                    f.write("    Existing output files are in preprocessed format (1.5mm spacing), NOT original image geometry.\n")
                    f.write("    Please fix the error and re-run to get outputs in original geometry.\n\n")

                print(f"Error: Task {task_id} failed: {e}")
                traceback.print_exc()
                print(f"Skipping task {task_id} and continuing with remaining tasks...")
                continue

        print(f"Finished in {time.time() - start:.2f}s")
