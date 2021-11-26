# QRGD
Repository for the implementation of "Distributed Principal Component Analysis with Limited Communication" (Alimisis et al., NeurIPS 2021). 

Parts of this code were originally based on code from "Communication-Efficient Distributed PCA by Riemannian Optimization" (Huang and Pan, ICML 2020).


## Usage

runQPCA.m is the main file, and is set up to run experiments comparing Riemannian gradient quantization against three other benchmark methods:

> Full-precision Riemannian gradient descent: Riemannian gradient descent, performed with the vectors communicated at full (64-bit) precision.

> Euclidean gradient difference quantization: the naive approach to quantizing Riemannian gradient descent. Euclidean gradients are quantized and averaged before being projected to Riemannian gradients and used to take a step. 
To improve performance, rather than quantizing Euclidean gradients directly, we quantize the difference between the current local gradient and the previous local gradient, at each node. Since these differences are generally smaller than the gradients themselves, we expect this quantization to introduce lower error.

> Quantized power iteration.

The first two benchmark methods, as well as our own method QRGD, are in the QPCA.m file, with options for quantization style and number of bits in the main method. 
Quantized power iteration is in the QPI.m file.

Four test datasets are also included: 

Human Activity from the MATLAB Statistics and Machine Learning Toolbox, 
Mice Protein Expression, Spambase, and Libras Movement from the UCI Machine Learning Repository.
