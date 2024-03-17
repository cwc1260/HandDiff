# HandDiff: 3D Hand Pose Estimation with Diffusion on Image-Point Cloud

Wencan Cheng, Hao Tang, Luc Van Gool and Jong Hwan Ko

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024

---
## Prerequisities
Our model is trained and tested under:
* Python 3.6.9
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.9.0)
* scipy
* tqdm
* Pillow
* yaml
* json
* cv2
* pycocotools

1. Prepare dataset 

    please download the NYU Hand dataset

2. Install PointNet++ CUDA operations

    follow the instructions in the './train_eval/pointnet2' for installation 

3. Evaluate

    set the "--test_path" paramter in the ```test_nyu.sh ``` as the path saved the generated testing set

    execute ``` sh test_nyu.sh```

    we provided the pre-trained models ('./results/nyu_handdiff_500iters_com/best_model.pth') for NYU

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

    set the "--dataset_path" paramter in the ```train_nyu.sh ``` as the path saved the generated traning and testing set respectively

    execute ``` sh train_nyu.sh```

If you find our code useful for your research, please cite our paper
```
@inproceedings{cheng2021handdiff,
  title={HandDiff: 3D Hand Pose Estimation with Diffusion on Image-Point Cloud},
  author={Cheng, Wencan, Hao Tang, Luc Van Gool and Ko, Jong Hwan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgement

We thank [repo](https://github.com/PengfeiRen96/IPNet) for the image-point cloud framework.


