#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import torch
import copy

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.config import CfgNode as _CfgNode

logger = logging.getLogger("detectron2")


class CN(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.

    .. automethod:: clone
    .. automethod:: freeze
    .. automethod:: defrost
    .. automethod:: is_frozen
    .. automethod:: load_yaml_with_base
    .. automethod:: merge_from_list
    .. automethod:: merge_from_other_cfg
    """

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        """
        Load content from the given config file and merge it into self.

        Args:
            cfg_filename: config filename
            allow_unsafe: allow unsafe yaml syntax
        """
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        from .defaults import _C

        latest_ver = _C.VERSION
        assert (
            latest_ver == self.VERSION
        ), "CfgNode.merge_from_file is only allowed on a config object of latest version!"

        logger = logging.getLogger(__name__)

        loaded_ver = loaded_cfg.get("VERSION", None)
        if loaded_ver is None:
            from .compat import guess_version

            loaded_ver = guess_version(loaded_cfg, cfg_filename)
        assert loaded_ver <= self.VERSION, "Cannot merge a v{} config into a v{} config.".format(
            loaded_ver, self.VERSION
        )

        if loaded_ver == self.VERSION:
            self.merge_from_other_cfg(loaded_cfg)
        else:
            # compat.py needs to import CfgNode
            from .compat import upgrade_config, downgrade_config

            logger.warning(
                "Loading an old v{} config file '{}' by automatically upgrading to v{}. "
                "See docs/CHANGELOG.md for instructions to update your files.".format(
                    loaded_ver, cfg_filename, self.VERSION
                )
            )
            # To convert, first obtain a full config at an old version
            old_self = downgrade_config(self, to_version=loaded_ver)
            old_self.merge_from_other_cfg(loaded_cfg)
            new_config = upgrade_config(old_self)
            self.clear()
            self.update(new_config)

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


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


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    cfg.model.roi_heads.num_classes = 10
    cfg.model.roi_heads.batch_size_per_image = 128

    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # cfg.dataloader.train.dataset.names = "coco_trash_train"
    # cfg.dataloader.test.dataset.names = "coco_trash_test"
    # print(cfg.dataloader.train)
    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    cfg.DATASETS = CN()
    # List of the dataset names for training. Must be registered in DatasetCatalog
    # Samples from these datasets will be merged and used as one dataset.
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    # List of the pre-computed proposal files for training, which must be consistent
    # with datasets listed in DATASETS.TRAIN.
    cfg.DATASETS.PROPOSAL_FILES_TRAIN = ()
    # Number of top scoring precomputed proposals to keep for training
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
    # List of the dataset names for testing. Must be registered in DatasetCatalog
    cfg.DATASETS.TEST = ('coco_trash_test',)
    # List of the pre-computed proposal files for test, which must be consistent
    # with datasets listed in DATASETS.TEST.
    cfg.DATASETS.PROPOSAL_FILES_TEST = ()
    # Number of top scoring precomputed proposals to keep for test
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000

    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    cfg.DATALOADER = CN()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    # Options: TrainingSampler, RepeatFactorTrainingSampler
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    # Repeat threshold for RepeatFactorTrainingSampler
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0
    # Tf True, when working on datasets that have instance annotations, the
    # training dataloader will filter out images without associated annotations
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.SOLVER = CN()
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.MODEL = CN()
    cfg.MODEL.LOAD_PROPOSALS = False
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False

    cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1


    for i in cfg.keys():
        print(i, cfg[i])
        print(" ")
    # -----------------------------------------------------------------------------

    # train_loader = instantiate(cfg.dataloader.train)
    train_loader = build_detection_train_loader(
        cfg, mapper=MyMapper, sampler=None
        )

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    # print(args.config_file)
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
