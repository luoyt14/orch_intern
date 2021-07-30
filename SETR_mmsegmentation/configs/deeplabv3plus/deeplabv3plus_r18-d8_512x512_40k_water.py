_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_single.py', '../_base_/datasets/pascal_voc12_water.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        num_classes=2,
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ), 
    auxiliary_head=dict(num_classes=2,in_channels=256, channels=64))
