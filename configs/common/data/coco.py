from omegaconf import OmegaConf
import copy
import torch

import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    DatasetCatalog,
    MetadataCatalog
)
from detectron2.evaluation import COCOEvaluator


def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)

    return dataset_dict


def TestMapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    dataset_dict['image'] = image

    return dataset_dict

# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '/data/dataset/upstage/dataset/train.json', '/data/dataset/upstage/dataset/')
except AssertionError:
    print("aaaa")
    pass

try:
    register_coco_instances('coco_trash_test', {}, '/data/dataset/upstage/dataset/test.json', '/data/dataset/upstage/dataset/')
except AssertionError:
    print("aaaa")
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_trash_train"),
    # mapper=L(DatasetMapper)(
    #     is_train=True,
    #     augmentations=[
    #         L(T.ResizeShortestEdge)(
    #             short_edge_length=(640, 672, 704, 736, 768, 800),
    #             sample_style="choice",
    #             max_size=1333,
    #         ),
    #         L(T.RandomFlip)(horizontal=True),
    #     ],
    #     image_format="BGR",
    #     use_instance_mask=True,
    # ),
    mapper=MyMapper,
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_trash_test", filter_empty=False),
    # mapper=L(DatasetMapper)(
    #     is_train=False,
    #     augmentations=[
    #         L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
    #     ],
    #     image_format="${...train.mapper.image_format}",
    # ),
    mapper=TestMapper,
    num_workers=4,
)

print("dataloader", dataloader)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
