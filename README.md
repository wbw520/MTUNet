```
python train.py --use_slot false --fsl false --drop_dim true --lr 0.001 --batch_size 256 --data_root ../../../wangbowen/data/mini_imagenet/
```

```
python train.py --use_slot true --fsl false --drop_dim true --lr 0.0001 --batch_size 256 --data_root ../../../wangbowen/data/mini_imagenet/
```

```
python train.py --use_slot true --fsl true --drop_dim false --lr 0.0001 --batch_size 1 --data_root ../../../wangbowen/data/mini_imagenet/
```