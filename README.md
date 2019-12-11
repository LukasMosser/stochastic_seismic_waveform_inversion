Adapted by Jonas Mendonça Targino

# Stochastic seismic waveform inversion using generative adversarial networks as a geological prior

Authors: [Lukas Mosser](https://twitter.com/porestar), [Olivier Dubrule](https://www.imperial.ac.uk/people/o.dubrule), [Martin J. Blunt](https://www.imperial.ac.uk/people/m.blunt) 

[Pytorch](https://pytorch.org) implementation of [Stochastic seismic waveform inversion using generative adversarial networks as a geological prior](https://arxiv.org/abs/1806.03720)

## Model Architecture

The model architecture consists of two parts:  
 - the generative adversarial network (implemented in [Pytorch](https://pytorch.org))   
 - the acoustic wave equation forward solver implemented in  ([Devito](https://www.opesci.org/devito)).  
The coupling between the two defines a fully differentiable computational graph.

## Movie representation of samples from the prior

<img src="https://github.com/LukasMosser/stochastic_seismic_waveform_inversion/raw/master/results/animations/movie_prior.gif" width="400">

## Movie representation of samples from the posterior (27 sources)

<img src="https://github.com/LukasMosser/stochastic_seismic_waveform_inversion/raw/master/results/animations/movie_posterior.gif" width="400">

## Usage

To perform the inversion using the available pre-trained generator network use ``` apps/main_paper_version_revised.py ```  
(Sorry for the long name, but older versions of the code were kept for reference purposes)
  
## Trained Models
Pre-trained models are available in the  [checkpoints](checkpoints/) directory.

## Results and Data

The resulting datasets are available in this [Google Drive](https://drive.google.com/drive/folders/1xLkLwDxAGVmfz-o2DzImgr8fP0fQNHW4?usp=sharing)  
 
Each run was made reproducible by setting the run-number = seed command-line argument.  
Computations were performed on Imperial College CX1 supercomputing facilities.  
Total duration: 12 hours wall-time on 32-core nodes ~ 50 nodes simultaneously.  

## Figures from paper

The figures from the paper can be reproduced using ```notebooks/Paper_Figures.ipynb```.  
All figures are located in ```results/figures```

## Devito Optimizations

The library used to represent the forward solver has a number of optimizations that allow it to parallelize across
cores and nodes using MPI.  
We suggest the following environment variables be set to maximize for performance:

```
DEVITO_OPENMP="1";
DEVITO_DLE="advanced"
DEVITO_LOGGING="INFO"
DEVITO_ARCH="gcc"
```

An example bash script used to perform the numerical computations on Imperial's CX1 cluster can be found in ```scripts/cluster_run.sh```

## Citing

```
@article{mosser2018stochastic,
  title={Stochastic seismic waveform inversion using generative adversarial networks as a geological prior},
  author={Mosser, Lukas and Dubrule, Olivier and Blunt, Martin J},
  journal={arXiv preprint arXiv:1806.03720},
  year={2018}
}
```

## Acknowledgements

The author would like to acknolwedge the developers of the [Devito](https://www.opesci.org/devito/).  
If you use their software, please acknowledge them in your references.  
O. Dubrule would like to thank Total for seconding him as a visiting professor at Imperial College London.

## License

[MIT](LICENSE)
