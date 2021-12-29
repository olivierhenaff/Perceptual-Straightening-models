# Perceptual straightening models 

### Introduction ###

This repository provides a PyTorch implementation of the hierarchical model described in "Perceptual straightening of natural videos" by Henaff, Goris & Simoncelli (https://www.olivierhenaff.com/content/perceptual-straightening.pdf). The model consists of a two-stage nonlinear decomposition which mimics the initial stages of biological visual processing. 
- The first stage (LGN) uses center-surround filtering followed by luminance and contrast normalization (as in Berardino, NeurIPS 2017). 
- The second stage (V1) uses oriented, band-pass filtering follwed by non-linear pooling. 

### Usage ###

- Run main.py to evaluate all stages of the model on all sequences in our dataset. 
- Run plot.py to visualize the average straightening across all sequences. This is similar to the results of figure 6b in the paper (which weights the average by the number of subjects tested for each sequence).

### Reference ###

The models presented here were developped for the following paper:

```
@article{henaff2019perceptual,
  title={Perceptual straightening of natural videos},
  author={H{\'e}naff, Olivier J and Goris, Robbe LT and Simoncelli, Eero P},
  journal={Nature neuroscience},
  volume={22},
  number={6},
  pages={984--991},
  year={2019},
  publisher={Nature Publishing Group}
}
```

If you find them to be useful for your research, please consider citing it. 
