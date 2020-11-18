## Model Structure
![Structure Figure](figs/overall.png)

## Usage

##### Data Set
Download miniImageNet from [miniImageNet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)

Download tieredImageNet from [tiered-ImageNet](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)

Download cifarFS from [cifarFS](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)

set all dataset under "../FSL_data", for the same split setting run the following command:

```
python data/tiered_imagenet.py --data "../FSL_data/tiered-imagenet/"
python data/cifarfs.py --data "../FSL_data/cifar100/" --split "../FSL_data""

```

##### Training for backbone
```
python train_base.py --dataset miniImageNet --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_base.py --dataset miniImageNet --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
python train_base.py --dataset tiered-ImageNet --base_model resnet18 --channel 512 --num_classes 351 --data_root "../FSL_data"
python train_base.py --dataset tiered-ImageNet --base_model wideres --channel 640 --num_classes 351 --data_root "../FSL_data"
python train_base.py --dataset cifar100 --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_base.py --dataset cifar100 --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
```

##### Training for patterns
```
python train_scouter.py --random False --interval 10 --dataset miniImageNet --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_scouter.py --random False --interval 10 --dataset miniImageNet --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
python train_scouter.py --random False --interval 10 --dataset tiered-ImageNet --base_model resnet18 --channel 512 --num_classes 351 --data_root "../FSL_data"
python train_scouter.py --random False --interval 10 --dataset tiered-ImageNet --base_model wideres --channel 640 --num_classes 351 --data_root "../FSL_data"
python train_scouter.py --random False --interval 10 --dataset cifarfs --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_scouter.py --random False --interval 10 --dataset cifarfs --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
```

##### Training MTUNet
```
python train_fsl.py --random False --interval 10 --dataset miniImageNet --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_fsl.py --random False --interval 10 --dataset miniImageNet --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
python train_fsl.py --random False --interval 10 --dataset tiered-ImageNet --base_model resnet18 --channel 512 --num_classes 351 --data_root "../FSL_data"
python train_fsl.py --random False --interval 10 --dataset tiered-ImageNet --base_model wideres --channel 640 --num_classes 351 --data_root "../FSL_data"
python train_fsl.py --random False --interval 10 --dataset cifarfs --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
python train_fsl.py --random False --interval 10 --dataset cifarfs --base_model wideres --channel 640 --num_classes 64 --data_root "../FSL_data"
```

##### Test MTUNet
```
python test_fsl.py --random False --interval 10 --dataset miniImageNet --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
```

##### Visualization of MTUNet
```
python vis_fsl.py --random False --interval 10 --dataset miniImageNet --base_model resnet18 --channel 512 --num_classes 64 --data_root "../FSL_data"
```