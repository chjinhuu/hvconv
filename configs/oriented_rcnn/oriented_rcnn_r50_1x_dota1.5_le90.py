_base_ = './rotated_faster_rcnn_hbb_r50_fpn_1x_dota_oc'

# 1. 数据集的设置
dataset_type = 'DOTADataset'
classes = ('a', 'b', 'c', 'd', 'e')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/train/annotation_data',
        img_prefix='path/to/your/train/image_data'),
    val=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/val/annotation_data',
        img_prefix='path/to/your/val/image_data'),
    test=dict(
        type=dataset_type,

        # 注意将你的类名添加到字段 `classes`
        classes=classes,
        ann_file='path/to/your/test/annotation_data',
        img_prefix='path/to/your/test/image_data'))

# 2. 模型设置
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # 显式将所有 `num_classes` 字段从 15 重写为 5。。
        num_classes=15))