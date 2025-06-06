_base_ = ['../../../_base_/default_runtime.py']

# 12‐Keypoints: cabeça, ombros, braços, punhos, cintura, joelhos, pés
channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=list(range(12)),
    inference_channel=list(range(12))
)

# Conexões entre os 12 pontos (índices 0–11)
skeleton = [
    [0, 7], [7, 1], [7, 2], [1, 3], [2, 4],
    [3, 5], [4, 6], [7, 8], [7, 9], [8, 10],
    [9, 11]
]

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210,
         milestones=[170, 200], gamma=0.1, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1,
                        block='BOTTLENECK', num_blocks=(4,),
                        num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2,
                        block='BASIC', num_blocks=(4, 4),
                        num_channels=(48, 96)),
            stage3=dict(num_modules=4, num_branches=3,
                        block='BASIC', num_blocks=(4, 4, 4),
                        num_channels=(48, 96, 192)),
            stage4=dict(num_modules=3, num_branches=4,
                        block='BASIC', num_blocks=(4, 4, 4, 4),
                        num_channels=(48, 96, 192, 384))
        ),
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=12,            # ajustado para 12
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True)
)

# dataset & pipelines
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode='topdown',
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
                  'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True, pipeline=val_pipeline,
        num_joints=12  # força 12
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    metric='keypoints'
)
test_evaluator = val_evaluator

# visualization hook (sem passar channel_cfg nem skeleton)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends
)

work_dir = './work_dirs/td-hm_hrnet-w48_12kp-8xb32'
