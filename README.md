# maskrcnn.pytorch
mask rcnn implemented by pytorch

# requirements

* python 3.6
* pytorch 0.4

# Install

Download this repo, must have access to `https://github.com/louis-she/voc2012-dataset.torch`

```
git clone git@github.com:louis-she/maskrcnn.pytorch.git --recursive
```

Compile 2 torch extensions

```
cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
```

The value for `arch` parameter, refer to following table
| GPU | arch |
| --- | --- |
| TitanX | sm_52 |
| GTX 960M | sm_50 |
| GTX 1070 | sm_61 |
| GTX 1080 (Ti) | sm_61 |

# Demo

Use `jupyter notebook` to open the `demo.ipynb`, there should be no error to run through this notebook