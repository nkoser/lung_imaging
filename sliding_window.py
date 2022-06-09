from typing import Tuple, List

import numpy as np
import torch
from batchgenerators.augmentations.utils import pad_nd_image
from pkbar import pkbar
from scipy.ndimage import gaussian_filter


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def predict_3D(x: np.ndarray,
               step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
               use_gaussian: bool = False, pad_border_mode: str = "constant",
               pad_kwargs: dict = None, all_in_gpu: bool = False,
               verbose: bool = True, gen=None):
    return _internal_predict_3D_3Dconv_tiled(x=x, step_size=step_size, patch_size=patch_size, use_gaussian=use_gaussian,
                                             pad_border_mode=pad_border_mode, pad_kwargs=pad_kwargs,
                                             all_in_gpu=all_in_gpu, verbose=verbose, gen=gen)


def _internal_predict_3D_3Dconv_tiled(x: np.ndarray, step_size: float,
                                      patch_size: tuple, use_gaussian: bool,
                                      pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                      verbose: bool, gen=None):
    # better safe than sorry
    _gaussian_3d = None
    _patch_size_for_gaussian_3d = None
    assert len(x.shape) == 4, "x must be (c, x, y, z)"

    if verbose: print("step_size:", step_size)

    assert patch_size is not None, "patch_size cannot be None for tiled prediction"

    # for sliding window inference the image must at least be as large as the patch size. It does not matter
    # whether the shape is divisible by 2**num_pool as long as the patch size is
    data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
    data_shape = data.shape  # still c, x, y, z

    # compute the steps for sliding window
    steps = _compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    if verbose:
        print("data shape:", data_shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", c)
        print("number of tiles:", num_tiles)

    # we only need to compute that once. It can take a while to compute this due to the large sigma in
    # gaussian_filter
    if use_gaussian and num_tiles > 1:
        if _gaussian_3d is None or not all(
                [i == j for i, j in zip(patch_size, _patch_size_for_gaussian_3d)]):
            if verbose: print('computing Gaussian')
            gaussian_importance_map = _get_gaussian(patch_size, sigma_scale=1. / 8)

            _gaussian_3d = gaussian_importance_map
            _patch_size_for_gaussian_3d = patch_size
        else:
            if verbose: print("using precomputed Gaussian")
            gaussian_importance_map = _gaussian_3d

        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        # predict on cpu if cuda not available
        if torch.cuda.is_available():
            gaussian_importance_map = gaussian_importance_map.cuda(non_blocking=True)

    else:
        gaussian_importance_map = None

    if all_in_gpu:
        # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
        # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

        if use_gaussian and num_tiles > 1:
            # half precision for the outputs should be good enough. If the outputs here are half, the
            # gaussian_importance_map should be as well
            gaussian_importance_map = gaussian_importance_map.half()

            # make sure we did not round anything to 0
            gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                gaussian_importance_map != 0].min()

            add_for_nb_of_preds = gaussian_importance_map
        else:
            add_for_nb_of_preds = torch.ones(patch_size).cuda()

        if verbose: print("initializing result array (on GPU)")
        aggregated_results = torch.zeros([1] + list(data.shape[1:]), dtype=torch.half).cuda()

        if verbose: print("moving data to GPU")
        data = torch.from_numpy(data).cuda(non_blocking=True)

        if verbose: print("initializing result_numsamples (on GPU)")
        aggregated_nb_of_predictions = torch.zeros([1] + list(data.shape[1:]), dtype=torch.half).cuda()

    else:
        if use_gaussian and num_tiles > 1:
            add_for_nb_of_preds = _gaussian_3d
        else:
            add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
        aggregated_results = np.zeros([1] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([1] + list(data.shape[1:]), dtype=np.float32)
    patches = []
    org_patches = []
    iterate = num_tiles
    bar = pkbar.Pbar(name="Progress", target=iterate)
    count = 0
    with torch.set_grad_enabled(False):
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    patch = data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]
                    if gen is None:
                        patches.append(patch)
                    else:
                        bar.update(count)
                        count += 1
                        org_patches.append(patch)
                        patches.append(gen(patch))
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += patches[-1].squeeze(
                        0).detach().cpu().numpy()
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
    slicer = tuple(
        [slice(0, aggregated_results.shape[i]) for i in
         range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    print(slicer)
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
    return org_patches, patches, aggregated_results, aggregated_nb_of_predictions