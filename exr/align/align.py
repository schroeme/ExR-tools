"""
Volumetric alignment module is designed to facilitate the precise alignment of volumetric microscopy data, particularly for large volumes, which are characterized by having large Z-slices. The central function, **`volumetric_alignment`**, serves as the primary interface for users to perform alignment operations. It abstracts the complexities of the underlying alignment mechanisms and offers a simple, unified entry point for processing.
"""
import h5py
import tempfile
import queue
import multiprocessing
import numpy as np
from typing import Tuple, Optional, List

from exr.config import Config
from exr.io import nd2ToVol
from exr.utils import subtract_background_top_hat, subtract_background_rolling_ball

from bigstream.transform import apply_transform
from bigstream.align import affine_align

from exr.utils import configure_logger

logger = configure_logger('ExR-Tools')


def transform_ref_round(config, roi, bg_sub):
    r"""
    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param roi: Region of interest.
    :type roi: int
    :param bg_sub: Specifies the background subtraction method to be used. 
                   Can be "rolling_ball" or "top_hat". If not provided, no background 
                   subtraction will be applied.
    :type bg_sub: str, optional
    """

    logger.info(f"Transform ref round: Round:{config.ref_round},ROI:{roi}")

    for channel_ind, channel in enumerate(config.channel_names):
        try:
            ref_vol = nd2ToVol(config.nd2_path.format(
                config.ref_round, roi), channel)

            if channel == config.ref_channel and bg_sub == 'rolling_ball':
                ref_vol = subtract_background_rolling_ball(ref_vol)

            if channel == config.ref_channel and bg_sub == 'top_hat':
                ref_vol = subtract_background_top_hat(ref_vol)

            with h5py.File(config.h5_path.format(config.ref_round, roi), "a") as f:
                if channel in f.keys():
                    del f[channel]
                f.create_dataset(channel, ref_vol.shape,
                                 dtype=ref_vol.dtype, data=ref_vol)
        except Exception as e:
            logger.error(
                f"Error during transformation for  Ref Round, ROI: {roi}, Channel: {channel}, Error: {e}")
            raise


def execute_volumetric_alignment(config: Config,
                                 tasks_queue: multiprocessing.Queue,
                                 q_lock: multiprocessing.Lock) -> None:
    r"""
    For each volume in code_fov_pairs, finds the corresponding reference volume and performs alignment.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param tasks_queue: A multiprocessing queue containing tasks.
    :type tasks_queue: multiprocessing.Queue
    :param q_lock: A lock for synchronizing tasks queue access.
    :type q_lock: multiprocessing.Lock
    """

    import SimpleITK as sitk

    while True:  # Check for remaining task in the Queue

        try:
            with q_lock:
                round, roi = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:
            try:

                if round == config.ref_round:
                    transform_ref_round(config, roi)

                sitk.ProcessObject_SetGlobalWarningDisplay(False)

                logger.info(f"Aligning: round:{round},ROI:{roi}")

                fix_vol = nd2ToVol(config.nd2_path.format(
                    config.ref_round, roi), config.ref_channel)

                mov_vol = nd2ToVol(config.nd2_path.format(
                    round, roi), config.ref_channel)

                fix_vol_sitk = sitk.GetImageFromArray(fix_vol)
                fix_vol_sitk.SetSpacing(config.spacing)

                mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                mov_vol_sitk.SetSpacing(config.spacing)

                # Initialize transform using Center of Gravity
                initial_transform = sitk.CenteredTransformInitializer(
                    fix_vol_sitk, mov_vol_sitk,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

                # Apply the transform to the moving image
                mov_vol_sitk = sitk.Resample(
                    mov_vol_sitk, fix_vol_sitk, initial_transform, sitk.sitkLinear, 0.0, mov_vol_sitk.GetPixelID())

                # temp dicectory for the log files
                tmpdir_obj = tempfile.TemporaryDirectory()

                # Align
                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.SetLogToFile(False)
                elastixImageFilter.SetLogToConsole(True)
                elastixImageFilter.SetOutputDirectory(tmpdir_obj.name)

                elastixImageFilter.SetFixedImage(fix_vol_sitk)
                elastixImageFilter.SetMovingImage(mov_vol_sitk)

                # Translation across x, y, and z only
                parameter_map = sitk.GetDefaultParameterMap("translation")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]
                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedRecursiveImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingRecursiveImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["5"]
                parameter_map["FixedImagePyramidSchedule"] = [
                    "10 10 10 8 8 8 4 4 4 2 2 2 1 1 1"]
                elastixImageFilter.SetParameterMap(parameter_map)

                # Translation + rotation
                parameter_map = sitk.GetDefaultParameterMap("rigid")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]
                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedShrinkingImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingShrinkingImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["1"]
                parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
                parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
                elastixImageFilter.AddParameterMap(parameter_map)

                # Translation, rotation, scaling and shearing
                parameter_map = sitk.GetDefaultParameterMap("affine")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]

                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedShrinkingImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingShrinkingImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["1"]
                parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
                parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
                elastixImageFilter.AddParameterMap(parameter_map)

                elastixImageFilter.Execute()

                transform_map = elastixImageFilter.GetTransformParameterMap()

                for channel_ind, channel in enumerate(config.channel_names):

                    mov_vol = nd2ToVol(
                        config.nd2_path.format(round, roi), channel)
                    mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                    mov_vol_sitk.SetSpacing(config.spacing)

                    transformixImageFilter = sitk.TransformixImageFilter()
                    transformixImageFilter.SetMovingImage(mov_vol_sitk)
                    transformixImageFilter.SetTransformParameterMap(
                        elastixImageFilter.GetTransformParameterMap())
                    transformixImageFilter.LogToConsoleOff()
                    transformixImageFilter.Execute()

                    out = sitk.GetArrayFromImage(
                        transformixImageFilter.GetResultImage())
                    with h5py.File(config.h5_path.format(round, roi), "a") as f:
                        if channel in f.keys():
                            del f[channel]
                        f.create_dataset(channel, out.shape,
                                         dtype=out.dtype, data=out)

                tmpdir_obj.cleanup()
            except Exception as e:
                logger.error(
                    f"Error during alignment for round: {round}, ROI: {roi}, Error: {e}")
                raise


def execute_volumetric_alignment_bigstream(config: Config,
                                           tasks_queue: multiprocessing.Queue,
                                           q_lock: multiprocessing.Lock) -> None:
    r"""
    Executes volumetric alignment using BigStream for each round and ROI from the tasks queue.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param tasks_queue: A multiprocessing queue containing tasks. Each task is a tuple of (round, roi).
    :type tasks_queue: multiprocessing.Queue
    :param q_lock: A lock for synchronizing tasks queue access.
    :type q_lock: multiprocessing.Lock
    """

    while True:  # Check for remaining task in the Queue

        try:
            with q_lock:
                round, roi, bg_sub = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:
            try:

                if round == config.ref_round:
                    transform_ref_round(config, roi, bg_sub)
                    continue

                logger.info(f"aligning: Round:{round},ROI:{roi}")

                try:
                    with h5py.File(config.h5_path.format(config.ref_round, roi), "r") as f:
                        fix_vol = f[config.ref_channel][()]

                except Exception as e:
                    logger.error(
                        f"The refrence round for ROI:{roi} is not processed yet, {e}")

                mov_vol = nd2ToVol(config.nd2_path.format(
                    round, roi), config.ref_channel)

                if bg_sub == "rolling_ball":
                    mov_vol = subtract_background_rolling_ball(mov_vol)

                if bg_sub == "top_hat":
                    mov_vol = subtract_background_top_hat(mov_vol)

                affine_kwargs = {
                    'alignment_spacing': 0.5,
                    'shrink_factors': (10, 8, 4, 2, 1),
                    'smooth_sigmas': (8., 4., 4., 2., 1.),
                    'optimizer_args': {
                        'learningRate': 0.25,
                        'minStep': 0.,
                        'numberOfIterations': 400,
                    },
                }

                affine = affine_align(
                    fix_vol, mov_vol,
                    np.array(config.spacing), np.array(config.spacing),
                    **affine_kwargs,
                )

                for channel_ind, channel in enumerate(config.channel_names):

                    mov_vol = nd2ToVol(
                        config.nd2_path.format(round, roi), channel)

                    # apply affine only
                    aligned_vol = apply_transform(
                        fix_vol, mov_vol,
                        np.array(config.spacing), np.array(config.spacing),
                        transform_list=[affine,],
                    )

                    with h5py.File(config.h5_path.format(round, roi), "a") as f:
                        if channel in f.keys():
                            del f[channel]
                        f.create_dataset(channel, aligned_vol.shape,
                                         dtype=aligned_vol.dtype, data=aligned_vol)

            except Exception as e:
                logger.error(
                    f"Error during alignment for round: {round}, ROI: {roi}, Error: {e}")
                raise


'''
# TODO limit itk multithreading
'''


def volumetric_alignment(config: Config,
                         round_roi_pairs: Optional[List[Tuple[int, int]]] = None,
                         parallel_processes: int = 1,
                         method: Optional[str] = None,
                         bg_sub: Optional[str] = '') -> None:
    r"""
    Parallel processing support for alignment function.

    :param config: Configuration options. This should be an instance of the Config class.
    :type config: Config
    :param round_roi_pairs: A list of tuples, where each tuple is a (round, roi) pair. If None, uses all rounds and roi pairs from the config.
    :type round_roi_pairs: List[Tuple[int, int]], optional
    :param parallel_processes: The number of processes to use for parallel processing. Default is 1, which means no parallel processing.
    :type parallel_processes: int, optional
    :param method: The method to use for alignment. If 'bigstream', uses the 'execute_volumetric_alignment_bigstream' function. Otherwise, uses the 'execute_volumetric_alignment' function.
    :type method: str, optional
    :param bg_sub: Specifies the background subtraction method to be used. 
                   Can be "rolling_ball" or "top_hat". If not provided, no background 
                   subtraction will be applied.
    :type bg_sub: str, optional
    """

    child_processes = []
    tasks_queue = multiprocessing.Queue()
    q_lock = multiprocessing.Lock()

    if not round_roi_pairs:
        round_roi_pairs = [[round_val, roi_val]
                           for round_val in config.rounds for roi_val in config.rois]

    for round, roi in round_roi_pairs:
        tasks_queue.put((round, roi, bg_sub))

    for w in range(int(parallel_processes)):
        try:

            if method == 'bigstream':
                p = multiprocessing.Process(
                    target=execute_volumetric_alignment_bigstream, args=(config, tasks_queue, q_lock))
            else:
                p = multiprocessing.Process(
                    target=execute_volumetric_alignment, args=(config, tasks_queue, q_lock))

            child_processes.append(p)
            p.start()
        except Exception as e:
            logger.error(
                f"Error starting process for Round: {round}, ROI: {roi}, Error: {e}")

    for p in child_processes:
        p.join()


