import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np

from cads.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps


def dice_per_label(seg_a, seg_b, labelmap):
    """
    Dice per structure between two label maps of the same task.
    Structures that are empty in both label maps get Dice = NaN.
    """
    results = []
    for label_id, name in sorted(labelmap.items()):
        if label_id == 0:
            continue
        a = seg_a == label_id
        b = seg_b == label_id
        n_a, n_b = int(a.sum()), int(b.sum())
        if n_a + n_b == 0:
            dice = float('nan')
        else:
            dice = 2.0 * int(np.logical_and(a, b).sum()) / (n_a + n_b)
        results.append({'label': label_id, 'structure': name, 'voxels_a': n_a, 'voxels_b': n_b, 'dice': dice})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare two CADS segmentations of the same task (e.g. CPU vs GPU run) as Dice per structure.")
    parser.add_argument("-a", dest="seg_a", required=True, help="First segmentation (e.g. *_part_558.nii.gz from a CPU run)")
    parser.add_argument("-b", dest="seg_b", required=True, help="Second segmentation of the same case and task")
    parser.add_argument("-task", dest="task_id", type=int, required=True, choices=sorted(map_taskid_to_labelmaps.keys()),
                        help="Task id of both segmentations (551-559)")
    parser.add_argument("-out", dest="output_csv", default=None, help="Optional csv file for the per-structure results")
    parser.add_argument("-min_dice", type=float, default=None,
                        help="Exit with code 1 if any structure present in either segmentation has a Dice below this value")
    args = parser.parse_args()

    img_a, img_b = nib.load(args.seg_a), nib.load(args.seg_b)
    if img_a.shape != img_b.shape:
        sys.exit(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")
    if not np.allclose(img_a.affine, img_b.affine, atol=1e-4):
        print("Warning: affines differ between the two segmentations")
    seg_a = np.asanyarray(img_a.dataobj).astype(np.int16)
    seg_b = np.asanyarray(img_b.dataobj).astype(np.int16)

    results = dice_per_label(seg_a, seg_b, map_taskid_to_labelmaps[args.task_id])
    present = [r for r in results if not np.isnan(r['dice'])]
    identical_voxels = float((seg_a == seg_b).mean())

    print(f"{'label':>5}  {'structure':<40} {'voxels_a':>10} {'voxels_b':>10} {'dice':>8}")
    for r in results:
        dice = 'n/a' if np.isnan(r['dice']) else f"{r['dice']:.4f}"
        print(f"{r['label']:>5}  {r['structure']:<40} {r['voxels_a']:>10} {r['voxels_b']:>10} {dice:>8}")
    if present:
        dices = [r['dice'] for r in present]
        print(f"Structures present: {len(present)}/{len(results)} | Dice min={min(dices):.4f} mean={np.mean(dices):.4f} | "
              f"identical voxels: {identical_voxels * 100:.4f}%")
    else:
        print("No structures present in either segmentation")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['label', 'structure', 'voxels_a', 'voxels_b', 'dice'])
            writer.writeheader()
            writer.writerows(results)

    if args.min_dice is not None and any(r['dice'] < args.min_dice for r in present):
        sys.exit(1)


if __name__ == "__main__":
    main()
