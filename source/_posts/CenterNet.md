---
title: CenterNet
date: 2022-11-30 12:54:19
tags: 
- MLDA
- Deep Learning
- Object Detection
categories:
- MLDA
---
- [CenterNet reproduction](#centernet-reproduction)
  - [Problems](#problems)
    - [CUDA version](#cuda-version)
    - [COCOapi](#cocoapi)

# CenterNet reproduction
## Problems

### CUDA version
> Because the Pytorch version used in this project is 0.4.1, which only support at most CUDA9.0 (current is 10.1 in our server), building the `DCNv2` directly, nvcc will generate a different dynamic library. When we run the code, this will lead to an import error.

There are two possible ways to solve this: change the pytorch version or change the cuda version. 

The first solution seems safer. However, torch.utils.ffi is a dependency of the following steps and it is only provided in pytorch<=0.4.1 so that we cannot change it.

Thus, we choose the second solution, to install another cuda. To make sure all the processes are safe, we create a new docker with corresponding environment.

### COCOapi
> When building the COCOapi, the `setup.py` will search for the latest matplotlib to install, which will be incompatible with the python3.6.

To solve this problem, modify the `setup.py` in `/path/to/cocoapi/PythonAPI` from
```python
setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
```
to
```python
setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib==2.1.0' # this line changed
    ],
    version='2.0',
    ext_modules= ext_modules
)
```

