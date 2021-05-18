import tensorflow as tf

import numpy as np
class GetModel():
    """
    Define the model using a backbone from tf.keras.applications and add custom heads.
    """
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name
        self.applications = tf.keras.applications
        
    def _set_backbone(self, *args, **kwargs): # input_shape, include_top=False, weights='imagenet', pooling='avg'
        attributes = getattr(self.applications, self.backbone_name)
        if hasattr(self.applications, self.backbone_name) and callable(attributes):
            return attributes(*args, **kwargs)

    def add_head(_set_backbone, n_classes: int, dropout: float, nodes: int, *args, **kwargs):
        backbone = _set_backbone(*args, **kwargs)
        x = tf.keras.layers.BatchNormalization()(backbone.output)
        x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Dense(nodes, activation='relu')
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)

        return tf.keras.Model(inputs=backbone.inputs, outputs=output)