# Region Ensemble Network: Improving Convolutional Network for Hand Pose Estimation
By Hengkai Guo (Updated on April 8, 2017)

## Description
This is the project of work [Region Ensemble Network: Improving Convolutional Network for Hand Pose Estimation](https://arxiv.org/abs/1702.02447). Currently this repository only includes the prediction results for comparison. More details will be released in the future.

## Results
Here we provide the testing results of basic network (`results/dataset_basic.txt`) and region ensemble network (`results/dataset_ren_4x6x6.txt`) for [ICVL](http://www.iis.ee.ic.ac.uk/~dtang/hand.html) dataset and [NYU](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) dataset in our paper. Also we provide the testing labels (`labels/dataset_test_label.txt`), computed centers (`labels/dataset_center.txt`) and corresponding image names (`labels/dataset_test_list.txt`).

For results and labels, each line is corresponding to one image, which has J x 3 numbers indicating (x, y, z) of J joint locations. The (x, y) are in pixels and z is in mm.

## Evaluation
Please use the Python scripts for evaluation, which requires numpy and matplotlib libraries. For example:
``` bash
$ python evaluation/compute_error.py icvl results/icvl_ren_4x6x6.txt
```

## Models
The models will be released soon.

## Citation
Please cite the paper in your publications if it helps your research:

```
@article{guo2017region,
  title={Region Ensemble Network: Improving Convolutional Network for Hand Pose Estimation},
  author={Guo, Hengkai and Wang, Guijin and Chen, Xinghao and Zhang, Cairong and Qiao, Fei and Yang, Huazhong},
  journal={arXiv preprint arXiv:1702.02447},
  year={2017}
}
```

## Feedback
Please email to `guohengkaighk@gmail.com` if you have any suggestions or questions.
