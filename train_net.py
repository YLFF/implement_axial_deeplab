#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""
DETECTRON2_DATASETS='/home/wangzhilin/work/code/project2021/datasets'
import os
import torch
from torchsummary import summary

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    #cfg.CUDNN_BENCHMARK=True
    cfg.merge_from_file('/home/wangzhilin/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml')
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.IMS_PER_BATCH = 8
    #cfg.MODEL.BACKBONE_NAME='build_axial_deeplab_backbone'
    cfg.MODEL.WEIGHTS='/home/wangzhilin/work/code/project2021/detectron2/models/Panoptic_deeplab_R52_DC5/model_final_23d03a.pkl' 
    cfg.OUTPUT_DIR = '/home/wangzhilin/work/code/project2021/detectron2/panopticseg/test/'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    args.num_gpus=1
    if args.eval_only:
        model = Trainer.build_model(cfg)
        #print(model)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus=1
    args.num_gpus=1
    args.dist_url='auto'
    args.eval_only=True

    
    print("Command Line Args:", args)
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
