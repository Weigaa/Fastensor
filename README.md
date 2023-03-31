# Introduction:

Fastensor is a file read/write interface for fast transfer of tensor between GPU and SSD. It is faster than Torch.load() and Torch.save() in any case.

We achieve up to 5.37x and 2.96x speedup ratios when using Fastensor for typical DNN model preservation and intermediate feature map offloading scenarios, respectively.



# Requirements:

You must have an NVIDIA GPU and NVMe SSD that can support GDS. Otherwise, the program may run in compatibility mode and not achieve the desired acceleration.

The software environment requirements are as follows:

CUDA 11.7.64
Pytorch 1.12.1
DALI 1.18.0
Numpy 1.23.1
Kvikio 22.08.0
Cupy 11.2.0

# Running method:

For model savings:

run: 

`cd Model Saving`

`python testmodel.py`

For model training with intermediate feature map transfer:

run: 

`cd Intermediate Feature Map Transfer`

run Bert model:

`cd Bert`

`python testbert_new.py`

run Vit model:

`cd vit`

`python testvit.py`







