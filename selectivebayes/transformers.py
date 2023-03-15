import pandas as pd
import seaborn as sns
import rdkit
import torch
import vina
import meeko
import pexpect
import pickle
import numpy as np
from scipy.stats import norm
from typing import Optional, Union, List
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.domain_reduction import DomainTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.target_space import TargetSpace
import sys
from contextlib import redirect_stdout

class UniformDomainReduction(DomainTransformer):
#domain reduction by reducing the domain size by a constant reduction_rate factor each iteration and increasing the domain size by factor increase_rate when
    #a duplicate is seen so at steady state, ~(1-reduction_rate)/(increase_rate-1) gives fraction of duplicates
    def __init__(self, vinainter, reduction_rate = 0.99, increase_rate = 1.10):
        self.vinainter = vinainter
        self.reduction_rate = reduction_rate
        self.increase_rate = increase_rate
        
    def initialize(self, target_space: TargetSpace):
        self.original_bounds = np.copy(target_space.bounds)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.original_width = self.original_bounds[:,1]-self.original_bounds[:,0]
        #set original prob
        self.prob = 1
        self.prev_seen = 0
    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}
    def transform(self, target_space: TargetSpace):
        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]
        #if the number of molecules seen has increased, increase prob by factor increase_rate, if prob is bigger than original_prob, set to original_prob
        if self.vinainter.seen>self.prev_seen:
            self.prob*=self.increase_rate
            self.prob = min(self.prob,1)
            self.prev_seen = self.vinainter.seen
            print("SEEN")
        else:
            #decrease prob by factor reduction_rate
            self.prob*=self.reduction_rate
        self.minimum_window = np.repeat(self.current_optimal[:,None],2,axis=1)
        self.minimum_window[:,1]+=self.prob/2*self.original_width
        self.minimum_window[:,0]-=self.prob/2*self.original_width
        difference_upper = self.minimum_window[:,1]-self.original_bounds[:,1]
        difference_upper[difference_upper<0]=0
        self.minimum_window[:,1]-=difference_upper
        self.minimum_window[:,0]-=difference_upper
        difference_lower = self.original_bounds[:,0]-self.minimum_window[:,0]
        difference_lower[difference_lower<0]=0
        self.minimum_window[:,0]+=difference_lower
        self.minimum_window[:,1]+=difference_lower

        return self._create_bounds(target_space.keys, self.minimum_window)

class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        prob = 0.2
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        self.prob = prob

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _update(self, target_space: TargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        prob = self.prob
        #make symmetric interval about current best with area prob
        #this could be optimised but is good enough for our purpose
        self.minimum_window = np.repeat(self.current_optimal[:,None],2,axis=1)
        for i in range(new_bounds.shape[0]):
            while norm.cdf(self.minimum_window[i,1])-norm.cdf(self.minimum_window[i,0])<prob:
                if self.minimum_window[i,1]<global_bounds[i,1]:
                    self.minimum_window[i,1]+=0.05
                if self.minimum_window[i,0]>global_bounds[i,0]:
                    self.minimum_window[i,0]-=0.05
        for i in range(new_bounds.shape[0]):
            if new_bounds[i,0]>self.minimum_window[i,0]:
                new_bounds[i,0]=self.minimum_window[i,0]
            if new_bounds[i,1]<self.minimum_window[i,1]:
                new_bounds[i,1]=self.minimum_window[i,1]
            if new_bounds[i,0]<global_bounds[i,0]:
                new_bounds[i,0]=global_bounds[i,0]
            if new_bounds[i,1]>global_bounds[i,1]:
                new_bounds[i,1]=global_bounds[i,1]

        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict:

        self._update(target_space)

        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)

class SimpleDomainReduction(DomainTransformer):
    #domain reduction by reducing the domain size by a constant reduction_rate factor each iteration and increasing the domain size by factor increase_rate when
    #a duplicate is seen so at steady state, ~(1-reduction_rate)/(increase_rate-1) gives fraction of duplicates
    def __init__(self, vinainter, reduction_rate = 0.99, increase_rate = 1.10):
        self.vinainter = vinainter
        self.reduction_rate = reduction_rate
        self.increase_rate = increase_rate
        
    def initialize(self, target_space: TargetSpace):
        self.original_bounds = np.copy(target_space.bounds)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        #set original prob
        self.original_prob = norm.cdf(self.original_bounds[0,1])-norm.cdf(self.original_bounds[0,0])
        self.prob = self.original_prob
        self.prev_seen = 0
    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}
    def transform(self, target_space: TargetSpace):
        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]
        #if the number of molecules seen has increased, increase prob by factor increase_rate, if prob is bigger than original_prob, set to original_prob
        if self.vinainter.seen>self.prev_seen:
            self.prob*=self.increase_rate
            self.prob = min(self.prob,self.original_prob)
            self.prev_seen = self.vinainter.seen
            print("SEEN")
        else:
            #decrease prob by factor reduction_rate
            self.prob*=self.reduction_rate
        global_bounds = self.original_bounds
        #make symmetric interval about current best with area prob
        #this could be optimised but is good enough for our purpose
        self.minimum_window = np.repeat(self.current_optimal[:,None],2,axis=1)
        for i in range(self.minimum_window.shape[0]):
            while norm.cdf(self.minimum_window[i,1])-norm.cdf(self.minimum_window[i,0])<self.prob:
                if self.minimum_window[i,1]<global_bounds[i,1]:
                    self.minimum_window[i,1]+=0.05
                if self.minimum_window[i,0]>global_bounds[i,0]:
                    self.minimum_window[i,0]-=0.05
        return self._create_bounds(target_space.keys, self.minimum_window)


