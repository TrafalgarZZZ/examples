# PyTorch分布式深度学习模型训练源码

支持两个数据集：
- ImageNet-ILSVRC 2012
- OpenImages

相关训练代码均存放在`./imagenet/`目录下
- 文件名包含`remote`的均为使用CacheSysAutoScaler组件的训练程序，不包含的则为冷启动/缓存预热场景
- 文件名包含`imagenet`或`openimages`分别是ImageNet和OpenImages数据集上的训练程序
- 文件名包含`elastic`的是弹性深度学习训练程序
- 文件名包含`breakdown`是会打印出具体的数据读取、数据预处理、模型训练时间分解时长的训练程序

各个训练程序可直接通过`train_*_<dataset-name>.sh`在容器化环境运行多机多卡分布式作业：

## PyTorch内核代码变更 

变更的代码位于`libs`目录下：
- torch/utils/data应当覆盖原始PyTorch内核代码的torch.utils.data包
- torchvision应当覆盖原始PyTorch内核代码的torchvision包

覆盖方式如下：

1. Docker Hub找到对应版本的PyTorch镜像，这里以`pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime`为例

```
$ docker run --name pytorch-original -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime bash
```

2. 找到torch和torchvision py包所在位置

```
$ pip show torch
Name: torch
...
Location: /opt/conda/lib/python3.7/site-packages
```

```
$ pip show torchvision
Name: torchvision
...
Location: /opt/conda/lib/python3.7/site-packages
```

3. 替换相关依赖

```
# 替换torch.utils.data包
$ cp ./libs/torch/utils/data/* /opt/conda/lib/python3.7/site-packages/torch/utils/data/

# 替换torchvision, torchvision中实际上仅增加了torchvision/datasets/remote_folder.py以及remote_vision.py，仅添加这部分即可
$ cp ./libs/torchvision/datasets/remote_*.py /opt/conda/lib/python3.7/site-packages/torchvision/datasets/
```

4. 安装其他依赖
```
$ pip install grpcio==1.37.1 grpcio-tools==1.37.1
```

5.退出但不关闭运行的Docker容器(`ctrl+p+q`)，执行docker commit生成新的Docker镜像

```
docker commit pytorch-original pytorch/pytorch:1.6.0-with-cache-sys-autoscaler
```

