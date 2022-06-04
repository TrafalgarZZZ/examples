# Modified PyTorch examples for Fluid system evaluation

Please follow imagenet/README.md for more details about distributed training.

不同数据集对应的训练Py源代码脚本不同：
- ImageNet-ILSVRC数据集请使用 `imagenet/main.py` (冷启动/缓存预热场景) 或 `imagenet/main_remote.py` (冷启动+Fluid场景)
- OpenImages-Subset数据集请使用 `imagenet/main.py` (冷启动/缓存预热场景) 或 `imagenet/main_remote_openimages.py` （冷启动+Fluid场景）
