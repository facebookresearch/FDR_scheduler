# PyTorch Implementation of FDR Scheduler

This library provides a PyTorch implementation of the learning-rate scheduler for stochastic gradient descent (SGD), based on the fluctuation-dissipation relation (FDR) described in the paper [1].

---

## Installation
```
pip install git+https://github.com/facebookresearch/FDR_scheduler
```

## Usage
This library provides a subclass of `Optimizer` that implements FDR scheduler. After installation, first import the scheduler
```python
from FDR_SGD import FDR_quencher
```
It is called quencher because the learning rate decreases (quenches) rather than increases (anneals) upon hitting equilibrium.

To use FDR scheduler, we set the optimizer as
```python
optimizer = FDR_quencher(model.parameters(), lr_init=0.1, momentum=0.0, dampening=0.0, weight_decay=0.001, t_adaptive=1000, X=0.01, Y=0.9)
```
where `lr_init`, `momentum`, `dampening`, and  `weight_decay` are initial learning rate, momentum, dampening, and weight decay of SGD, respectively. The descriptions of scheduling hyperparameters  `t_adaptive`, `X`, and `Y` are provided right below. Also please check out the file [`example.py`](./example.py).

**Note:** it is important to set weight decay to be nonzero; otherwise SGD sampling typically does not reach equilibrium. For additional information, please see the paper [1].

### Keyword Arguments
 - t_adaptive (int, optional, default=1000 SGD iterations): how frequently one checks proximity to equilibrium.
 - X (float, optional, default=0.01): how stringently one imposes the equilibrium condition. For example, the default value 0.01 means that the fluctuation-dissipation relation needs to be satisfied within one-percent error.
 - Y (float, optional, default=0.9): learning rate decreases by *(1-Y) once SGD sampling is deemed close to equilibrium. For example, the default value 0.9 implies the ten-fold decrease in the learning rate.

## License
FDR_scheduler is licensed under the MIT license found in the LICENSE file.

## References
[1] Sho Yaida, "Fluctuation-dissipation relations for stochastic gradient descent," 2019. [ICLR 2019](https://openreview.net/forum?id=SkNksoRctQ) [arxiv:1810.00004](https://arxiv.org/abs/1810.00004)

---

If you found this library useful, please consider citing
```
@inproceedings{yaida2018fluctuationdissipation,
      author         = "Sho Yaida",
      title          = "Fluctuation-dissipation relations for stochastic gradient descent",
      booktitle      = "International Conference on Learning Representations",
      year           = "2019",
      url            = "https://openreview.net/forum?id=SkNksoRctQ",
}
```

## Acknowledgments
I thank Daniel Adam Roberts for his help in cleaning the implementation of the FDR scheduler and our subsequent fun collaboration in theory and in practice.
