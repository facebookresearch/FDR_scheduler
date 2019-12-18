'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of the scheduler based on the fluctuation-dissipation relation described in:
    Sho Yaida, "Fluctuation-dissipation relations for stochastic gradient descent," ICLR 2019.
'''
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class FDR_quencher(Optimizer):
    '''
    Arguments:
        lr_init (float, optional, default=0.1): initial learning rate.
        
        momentum (float, optional, default=0.0): momentum.
        
        dampening (float, optional, default=0.0): dampening.
        
        weight_decay (float, optional, default=0.001): weight_decay.
            If weight_decay=0, SGD sampling typically does not reach equilibrium. Hence it should be set away from 0.
            
        t_adaptive (int, optional, default=1000 SGD iterations): how frequently one checks proximity to equilibrium.
        
        X (float, optional, default=0.01): how stringently one imposes the equilibrium condition.
            For example, the default value 0.01 means that the fluctuation-dissipation relation needs to be satisfied within one-percent error.
            
        Y (float, optional, default=0.9): learning rate decreases by *(1-Y) once SGD sampling is deemed close to equilibrium.
            For example, the default value 0.9 implies the ten-fold decrease in the learning rate.
    '''
    # Setting up hyperparameters
    def __init__(self, params, lr_init=0.1, momentum=0.0, dampening=0, weight_decay=0.001, t_adaptive=1000, X=0.01, Y=0.9):
        defaults = dict(lr=lr_init, momentum=momentum, dampening=dampening, weight_decay=weight_decay, t_adaptive=t_adaptive, X=X, Y=Y)
        super(FDR_quencher, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(FDR_quencher, self).__setstate__(state)
    
    # One SGD iteration
    def step(self,closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            
            # SGD hyperparameters for a group
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            
            # Allocating memory for observables in the first fluctuation-dissipation relation
            if 'OLs' not in group:
                # Observables in the first fluctuation-dissipation relation
                group['OLs'] = list()
                group['ORs'] = list()
                group['t'] = 0

            # Actual SGD step
            # F = -(gradient of loss)
            # v = momentum*v+(1-dampening)*F
            # theta = theta+lr*v where
            v_squared = 0.0
            OL=0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                F = -p.grad.data
                if weight_decay != 0:
                    F.add_(-weight_decay, theta)

                # Create velocity degrees of freedom
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    v = param_state['velocity'] = torch.zeros_like(theta)
                else:
                    v = param_state['velocity']

                # Compute observables
                v_squared += torch.norm(v)**2
                OL += -torch.dot(F.view(-1),theta.view(-1))

                # Update velocity and position
                v = v.mul_(momentum).add_(1.0-dampening,F)
                theta = theta.add_(lr,v)
                
                # Save current v update
                param_state['velocity'] = v
            
            # Record OL and OR
            OR = 0.5*lr*((1.0+momentum)/(1.0-dampening))*v_squared
            group['OLs'].append(OL.item())
            group['ORs'].append(OR.item())
            group['t'] += 1
            
            # Read time
            t_adaptive = group['t_adaptive']
            t = group['t']
        
            # FDR adaptive step
            if (t % t_adaptive)==0:

                # Fluctuation-dissipation observables
                OLs = group['OLs']
                ORs = group['ORs']
   
                # Compute half-running time average
                transient_cut=int(np.floor(0.5*t))
                OLbar=np.mean(OLs[transient_cut:t])
                ORbar=np.mean(ORs[transient_cut:t])
        
                # FDR adaptive scheduling hyperparameters
                X = group['X']
                Y = group['Y']

                # Adapt if sufficiently close to equilibrium
                cFDR = np.abs((OLbar/ORbar) - 1.0)
            
                # In order to peek into what is going on behind the scene, please uncomment the following
                #print('time=%d, cFDR=%.3f, lr=%.3f\n' % (group['t'], cFDR, group['lr']))
            
                # If close to equilibrium, decrease the learning rate
                if cFDR<X:
                
                    # Quench
                    group['lr'] *= (1.0-Y)
                    
                    # Purge running record of observables
                    group['OLs'] = list()
                    group['ORs'] = list()

                    # Reset time
                    group['t'] = 0
