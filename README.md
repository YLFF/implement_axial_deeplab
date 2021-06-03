# implement of axial deeplab

In implement of axial deeplab of panoptic segmentation task.
Based on [detectron2](https://github.com/facebookresearch/detectron2) and [Axial Deeplab](https://github.com/csrhddlam/axial-deeplab)(which didn't include the panoptic seg part).
For now, the model changed the res5 block from atrous conv to axial attention block.

To do:
- [ ] experiment result
- [ ] change of encoder stucture
- [ ] add local constraint module
