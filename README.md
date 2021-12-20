# Multi-task Learning of Order-Consistent Causal Graphs (NeuRIPs 2021)

- Authors: [Xinshi Chen](http://xinshi-chen.com/), [Haoran Sun](https://people.math.gatech.edu/~hsun349/), [Caleb Ellington](https://calebellington.com/), [Eric Xing](http://www.cs.cmu.edu/~epxing/), [Le Song](https://www.cc.gatech.edu/~lsong/)

- [Link to paper](https://arxiv.org/pdf/2111.02545.pdf)

- [Link to slides](http://xinshi-chen.com/papers/slides/neurips2021-multidag.pdf)

- [Link to NeurIPs 15-min presentation](https://neurips.cc/virtual/2021/poster/27045)


If you found this library useful in your research, please consider citing

```
@article{chen2021multi,
  title={Multi-task Learning of Order-Consistent Causal Graphs},
  author={Chen, Xinshi and Sun, Haoran and Ellington, Caleb and Xing, Eric and Song, Le},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{chen2021multi,
  title={Multi-task Learning of Order-Consistent Causal Graphs},
  author={Chen, Xinshi and Sun, Haoran and Ellington, Caleb and Xing, Eric and Song, Le},
  journal={arXiv preprint arXiv:2111.02545},
  year={2021}
}
```

# Reproduce Experiments

## Install the module
Please navigate to the root of this repository, and run the following command to install the `multidag` module.
```
pip install -e .
```

## Experiment 1: Simulations
Descriptions of implementation details coming soon.....

## Experiment 2: Recovering gene regulatory networks
We simulate single-cell gene regulatory networks with [SERGIO](https://github.com/PayamDiba/SERGIO) and recover them with MultiDAG. 
When the number of samples per network is limited, MultiDAG improves overall accuracy by sharing information between recovery tasks.

To reproduce our results, navigate to the `experiments/` folder and run `run_sergio.sh`
