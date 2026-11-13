import argparse
import gc
import os
import time
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

from cads.utils.libs import setup_nnunet_env

VALID_TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]


def main():
    parser = argparse.ArgumentParser(
        description="Export CADS nnU-Net models to OpenVINO IR (stored locally next to each checkpoint, "
                    "in fold_<x>/openvino/) for use with `-device openvino`.")
    parser.add_argument("-task", nargs="+", default=["all"],
                        help="Task ids to export (e.g. 552 553 556 558) or 'all' (default)")
    parser.add_argument("-license", choices=["reference", "research", "open"], default="reference",
                        help="Model variant whose weights are exported (same meaning as in predict_images.py)")
    parser.add_argument("-fp16", action="store_true",
                        help="Compress the IR weights to FP16 (written as *_fp16.xml; still computed in FP32 on CPU)")
    args = parser.parse_args()

    task_ids = VALID_TASKS if args.task == ["all"] else sorted({int(t) for t in args.task})
    if any(t not in VALID_TASKS for t in task_ids):
        parser.error(f"task ids must be in {VALID_TASKS}")

    setup_nnunet_env(args.license)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from cads.utils.inference import get_task_model_folder
    from cads.utils.libs import check_or_download_model_weights, get_model_weights_dir
    from cads.utils.openvino_backend import export_openvino

    model_folder = get_model_weights_dir(args.license)
    for task_id in task_ids:
        check_or_download_model_weights(task_id, args.license)
        task_model_folder = get_task_model_folder(task_id, model_folder)
        predictor = nnUNetPredictor(use_gaussian=True, use_mirroring=False, device=torch.device('cpu'),
                                    verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(task_model_folder, 'all', 'checkpoint_final.pth')
        t0 = time.time()
        for xml in export_openvino(predictor, task_model_folder, 'all', 'checkpoint_final.pth', fp16=args.fp16):
            size_mb = os.path.getsize(os.path.splitext(xml)[0] + '.bin') / 1024**2
            print(f"Task {task_id}: wrote {xml} ({size_mb:.0f} MB weights, {time.time() - t0:.1f} s)", flush=True)
        del predictor
        gc.collect()


if __name__ == "__main__":
    main()
