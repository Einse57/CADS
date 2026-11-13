"""
Optional OpenVINO backend for CADS inference.

Only the network forward pass is replaced: nnU-Net still does the preprocessing, sliding window,
Gaussian weighting, (optional) mirroring and export, and CADS still does its own pre- and post-processing.

Select it with `-device openvino` (OpenVINO CPU plugin) or `-device openvino:<OV_DEVICE>` (e.g. `openvino:GPU`),
or via the CADS_DEVICE environment variable.

The IR is looked up next to each checkpoint (`fold_<x>/openvino/<checkpoint>[_fp16].xml`, written by
cads/scripts/export_openvino.py). If it is missing, the network is converted in memory at start-up
(nothing is written to disk).

Environment variables:
    CADS_OV_IR=fp32|fp16       which exported IR to load (default fp32; fp16 = FP16-compressed weights)
    CADS_OV_PRECISION_HINT     OpenVINO INFERENCE_PRECISION_HINT; default 'f32' on CPU (so CPUs with bf16/AMX
                               do not silently switch to bf16), the plugin default on other devices.
                               Use 'default' to keep the plugin default everywhere.
"""
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch

OV_SUBDIR = 'openvino'


def parse_openvino_device(device):
    """Return the OpenVINO device name (e.g. 'CPU') if `device` requests OpenVINO, else None."""
    if device is None:
        return None
    s = str(device).strip()
    if s.lower() in ('auto', ''):
        s = os.environ.get('CADS_DEVICE', '').strip()
    if not s.lower().startswith('openvino'):
        return None
    _, _, ov_dev = s.partition(':')
    return ov_dev.strip().upper() or 'CPU'


def ir_path(fold_dir, checkpoint_name='checkpoint_final.pth', fp16=False):
    stem = Path(checkpoint_name).stem
    return Path(fold_dir) / OV_SUBDIR / f"{stem}{'_fp16' if fp16 else ''}.xml"


def _input_shape(predictor):
    patch_size = list(predictor.configuration_manager.patch_size)
    n_channels = len(predictor.dataset_json['channel_names'].keys())
    return [1, n_channels, *patch_size]


def _trace_shape(network, input_shape):
    """Smallest input the encoder accepts (2x the total stride per axis). The networks are fully
    convolutional, so tracing at this size keeps conversion memory low; the IR input is still fixed to
    the full patch size."""
    strides = getattr(getattr(network, 'encoder', None), 'strides', None)
    if not strides:
        return list(input_shape)
    total = [int(np.prod([s[i] if isinstance(s, (list, tuple)) else s for s in strides])) for i in range(len(input_shape) - 2)]
    return [*input_shape[:2], *[min(p, 2 * t) for p, t in zip(input_shape[2:], total)]]


def convert_network(network, input_shape):
    """Convert an nnU-Net network (in eval mode, deep supervision off) to an in-memory OpenVINO model
    with a fixed (1, C, *patch_size) input. nnU-Net's sliding window always feeds exactly one patch."""
    import openvino as ov
    network = network.to('cpu').eval()
    example = torch.zeros(_trace_shape(network, input_shape), dtype=torch.float32)
    with torch.no_grad():
        return ov.convert_model(network, example_input=example, input=[tuple(input_shape)])


def _load_state(predictor, params):
    net = predictor.network
    net = getattr(net, '_orig_mod', net)
    net.load_state_dict(params)
    return net


def fold_dirs(model_training_output_dir, folds):
    if isinstance(folds, (str, int)):
        folds = [folds]
    return [Path(model_training_output_dir) / (f'fold_{f}' if f != 'all' else 'fold_all') for f in folds]


class OpenVINONetwork(torch.nn.Module):
    """
    Drop-in replacement for predictor.network. nnU-Net calls network.load_state_dict(params) for each fold
    before running the sliding window; here that selects the OpenVINO model of that fold.
    Models are compiled on first use and released by release() after each prediction, so only one
    compiled model (with its patch-size activation buffers) is resident at a time.
    """

    def __init__(self, core, models, list_of_parameters, ov_device, config):
        super().__init__()
        self._core, self._models, self._params = core, list(models), list(list_of_parameters)
        self._ov_device, self._config = ov_device, dict(config)
        self._active = 0
        self._compiled = self._request = None
        self.forward_s = 0.0
        self.n_forward = 0

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for i, p in enumerate(self._params):
            if p is state_dict:
                return self._select(i)
        if len(self._models) == 1:
            return self._select(0)
        raise RuntimeError('OpenVINO backend: unknown parameter set (fold) requested')

    def _select(self, i):
        if i != self._active:
            self.release()
            self._active = i

    def compile(self):
        if self._request is None:
            self._compiled = self._core.compile_model(self._models[self._active], self._ov_device, self._config)
            self._request = self._compiled.create_infer_request()
        return self._compiled

    def release(self):
        self._request = self._compiled = None
        gc.collect()

    def forward(self, x):
        t0 = time.perf_counter()
        self.compile()
        arr = np.ascontiguousarray(x.detach().to('cpu', torch.float32).numpy())
        self._request.infer({0: arr})
        out = torch.from_numpy(np.array(self._request.get_output_tensor(0).data, copy=True))
        self.forward_s += time.perf_counter() - t0
        self.n_forward += 1
        return out.to(x.device)

    def reset_timer(self):
        self.forward_s, self.n_forward = 0.0, 0


def attach_openvino(predictor, model_training_output_dir, folds, checkpoint_name='checkpoint_final.pth', ov_device='CPU'):
    """Replace predictor.network (an initialized nnUNetPredictor) with an OpenVINONetwork."""
    import openvino as ov

    fp16 = os.environ.get('CADS_OV_IR', 'fp32').strip().lower() == 'fp16'
    hint = os.environ.get('CADS_OV_PRECISION_HINT', 'f32' if ov_device == 'CPU' else 'default').strip().lower()
    config = {'PERFORMANCE_HINT': 'LATENCY'}
    if hint != 'default':
        config['INFERENCE_PRECISION_HINT'] = hint

    core = ov.Core()
    input_shape = _input_shape(predictor)
    models = []
    for fold_dir, params in zip(fold_dirs(model_training_output_dir, folds), predictor.list_of_parameters):
        xml = ir_path(fold_dir, checkpoint_name, fp16)
        if xml.is_file():
            models.append(core.read_model(str(xml)))
            src = str(xml)
        else:
            _load_state(predictor, params)
            models.append(convert_network(predictor.network, input_shape))
            src = 'in-memory conversion (run cads/scripts/export_openvino.py to cache the IR)'
        print(f"[openvino] {fold_dir.name}: {src}")
    network = OpenVINONetwork(core, models, predictor.list_of_parameters, ov_device, config)
    try:
        precision = network.compile().get_property('INFERENCE_PRECISION_HINT')
        precision = precision.get_type_name() if hasattr(precision, 'get_type_name') else str(precision)
    except Exception:
        precision = hint
    network.release()
    print(f"[openvino] device={ov_device} inference precision={precision} IR={'fp16' if fp16 else 'fp32'} input={input_shape}")
    predictor.network = network
    # results stay on the CPU; nnU-Net only allows on-device accumulation for CUDA
    predictor.perform_everything_on_device = False
    return network


def export_openvino(predictor, model_training_output_dir, folds, checkpoint_name='checkpoint_final.pth', fp16=False):
    """Export every fold of an initialized nnUNetPredictor to OpenVINO IR next to its checkpoint."""
    import openvino as ov
    input_shape = _input_shape(predictor)
    written = []
    for fold_dir, params in zip(fold_dirs(model_training_output_dir, folds), predictor.list_of_parameters):
        _load_state(predictor, params)
        model = convert_network(predictor.network, input_shape)
        xml = ir_path(fold_dir, checkpoint_name, fp16)
        xml.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(model, str(xml), compress_to_fp16=fp16)
        written.append(xml)
    return written
