# straightening-hierarchy

PyTorch implementation of the hierarchical model described in "Perceptual straightening of natural videos" by Henaff, Goris & Simoncelli (2019). Run main.py to compute the average straightening across all sequences, for all stages in the model. This is similar to the results of figure 6b in the paper (which weights the average by the number of subjects tested for each sequence).  

Requires pytorch_fft from https://github.com/locuslab/pytorch_fft.
