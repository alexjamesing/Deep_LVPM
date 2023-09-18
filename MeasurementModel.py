#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:54:37 2022

@author: ing
"""

import os
current_path=os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(current_path)
import tensorflow as tf
from FactorLayer import FactorLayer
from zca_layer import zca_layer
import keras


#@keras.saving.register_keras_serializable()
class MeasurementModel(tf.keras.Model):
    
    """DPLS_PM_Model is used to instantiate a measurement model. This is a model
    for a particular data-view, which can then be used to identify associations
    with other data-views, to which it is connected in the global model. The inputs 
    to this model subclass are a base model, and parameters for the last layer 
    of the neural network. 
    Args:
    model_in: This is the base model for the Deep-PLS neural network. 
    kernel_regularizer: This is the weight regularisation that can be optionally added 
    to the projection layers of the Deep-PLS-PM model
    momentum: This parameter is used to stabilise covariance estimation in the 
    deep-pls-pm model.
    """
    
    def __init__(self, model_in, kernel_regularizer=None, epsilon=1e-6, momentum=0.95):
        
        super().__init__()
        
        self.kernel_regularizer=kernel_regularizer
        self.epsilon=epsilon
        self.momentum=momentum
        self.model_in=model_in
    
    def call(self,inputs):
        """ This function specifies the internal logic of the measurement model.
        The feed-forward function is always split into two parts. The model_in
        part of the network can be many different types of model. In contrast,
        the 'top_layer' will always be a Deep-PLS-PM layer.
        """
        
        x = self.model_in(inputs)
        out = self.top_layer(x)
        
        return out
    
    def global_build(self, tot_num, tot_dims, orthogonalisation):
        """ This function takes parameters that are global properties of the model,
        which are defined in the Structural Model, but which need to be specified at 
        the Measurement Model and layer-wise level. This includes the total number 
        of examples, the total number of dimensions to extract, and the method of
        orthogonalisation that is required.
        """
        self.tot_num = tot_num
        self.tot_dims = tot_dims
        self.orthogonalisation = orthogonalisation
        
        
        if orthogonalisation=='Moore-Penrose':
            self.top_layer=FactorLayer(self.kernel_regularizer,self.epsilon,self.momentum)
        elif orthogonalisation=='zca':
            self.top_layer=zca_layer(self.kernel_regularizer,self.epsilon,self.momentum)
        
        self.top_layer.global_build(self.tot_num,self.tot_dims)
        
    # def trial_build(self, tot_num, tot_dims, orthogonalisation, top_layer):
        
        
    #     self.tot_dims = tot_dims
    #     self.tot_num = tot_num
    #     self.orthogonalisation = orthogonalisation
    #     self.top_layer = top_layer
        
        
    # def get_config(self):
     
    #     config = super().get_config().copy()
        
    #     config.update({
    #         'kernel_regularizer': keras.saving.serialize_keras_object(self.kernel_regularizer),
    #         'momentum': self.momentum,
    #         'epsilon': self.epsilon,
    #         'model_in': tf.keras.saving.serialize_keras_object(self.model_in),
    #         'tot_num': self.tot_num,
    #         'tot_dims': self.tot_dims,
    #         'orthogonalisation': self.orthogonalisation,
    #         'top_layer': tf.keras.saving.serialize_keras_object(self.top_layer)
    #         })
     
    #     return config
     
    # @classmethod
    # def from_config(cls, config):
    
    #     cls(config["model_in"],keras.saving.deserialize_keras_object(config["kernel_regularizer"]),config["momentum"], 
    #         config["epsilon"])
        
        
    #     cls.trial_build(cls, config["tot_num"], config["tot_dims"], config["orthogonalisation"], config['top_layer'])
        
    #     return cls
    
        
# model = my_build_model_omics_singlemodel_mut(100, 100, 'mody')

# model.global_build(100,10,'Moore-Penrose')
        

# model
