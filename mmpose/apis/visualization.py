# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Union

import mmcv
import numpy as np
from mmengine.structures import InstanceData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def visualize(
    img: Union[np.ndarray, str],
    keypoints: np.ndarray,
    keypoint_score: np.ndarray = None,
    metainfo: Union[str, dict] = None,
    visualizer: PoseLocalVisualizer = None,
    show_kpt_idx: bool = False,
    skeleton_style: str = 'mmpose',
    show: bool = False,
    kpt_thr: float = 0.3,
):
    """Visualize 2d keypoints on an image.

    Args:
        img (str | np.ndarray): The image to be displayed.
        keypoints (np.ndarray): The keypoint to be displayed.
        keypoint_score (np.ndarray): The score of each keypoint.
        metainfo (str | dict): The metainfo of dataset.
        visualizer (PoseLocalVisualizer): The visualizer.
        show_kpt_idx (bool): Whether to show the index of keypoints.
        skeleton_style (str): Skeleton style. Options are 'mmpose' and
            'openpose'.
        show (bool): Whether to show the image.
        wait_time (int): Value of waitKey param.
        kpt_thr (float): Keypoint threshold.
    """
    assert skeleton_style in [
        'mmpose', 'openpose'
    ], (f'Only support skeleton style in {["mmpose", "openpose"]}, ')

    if visualizer is None:
        visualizer = PoseLocalVisualizer()
    else:
        visualizer = deepcopy(visualizer)

    if isinstance(metainfo, str):
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
    elif isinstance(metainfo, dict):
        # ``metainfo`` may already be parsed by ``init_model``. Only parse
        # again when it contains raw keys such as ``keypoint_info``.
        if 'keypoint_info' in metainfo:
            metainfo = parse_pose_metainfo(metainfo)

    if metainfo is not None:
        visualizer.set_dataset_meta(metainfo, skeleton_style=skeleton_style)

    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if keypoint_score is None:
        keypoint_score = np.ones(keypoints.shape[0])

    tmp_instances = InstanceData()
    tmp_instances.keypoints = keypoints
    tmp_instances.keypoint_score = keypoint_score

    tmp_datasample = PoseDataSample()
    tmp_datasample.pred_instances = tmp_instances

    visualizer.add_datasample(
        'visualization',
        img,
        tmp_datasample,
        show_kpt_idx=show_kpt_idx,
        skeleton_style=skeleton_style,
        show=show,
        wait_time=0,
        kpt_thr=kpt_thr)

    return visualizer.get_image()


def vis_pose_result(
    model,
    img: Union[np.ndarray, str],
    result: List[PoseDataSample],
    radius: int = 4,
    thickness: int = 1,
    kpt_score_thr: float = 0.3,
    show: bool = False,
    draw_heatmap: bool = False,
    alpha: float = 1.0,
    skeleton_style: str = 'mmpose',
) -> np.ndarray:
    """A compatibility wrapper of the deprecated ``vis_pose_result`` API.

    Args:
        model: The pose estimator model. Only ``model.dataset_meta`` is used.
        img (str | np.ndarray): Image file or array to draw.
        result (list[PoseDataSample]): Inference results from
            :func:`inference_topdown`.
        radius (int): Keypoint radius. Defaults to 4.
        thickness (int): Link thickness. Defaults to 1.
        kpt_score_thr (float): Threshold of keypoint scores. Defaults to 0.3.
        show (bool): Whether to show the image. Defaults to False.
        draw_heatmap (bool): Unused. For compatibility only.
        alpha (float): Transparency of the drawn results. Defaults to 1.0.
        skeleton_style (str): Skeleton style. Defaults to ``'mmpose'``.

    Returns:
        np.ndarray: The visualized image in ``RGB`` format.
    """

    if not isinstance(result, list):
        result = [result]

    keypoints = np.stack([r.pred_instances.keypoints[0] for r in result])
    scores = np.stack([r.pred_instances.keypoint_scores[0] for r in result])

    visualizer = PoseLocalVisualizer(
        radius=radius, line_width=thickness, alpha=alpha)

    return visualize(
        img,
        keypoints,
        scores,
        metainfo=getattr(model, 'dataset_meta', None),
        visualizer=visualizer,
        show_kpt_idx=False,
        skeleton_style=skeleton_style,
        show=show,
        kpt_thr=kpt_score_thr,
    )
