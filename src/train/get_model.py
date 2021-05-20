import tensorflow as tf

class GetModel:
    """
    Define the model using a backbone from tf.keras.applications and add a custom top (head).
    """
    def __init__(self):
        
        self.applications = tf.keras.applications
        self.model = None
        self.include_top = None
        
    def set_backbone(self, backbone_name: str, include_top=True, **kwargs):
        """
        Sets the model backbone. Can be chained with 'add_top()' only when 'include_top=False'.
        Args:
            backbone_name (str): The name of the backbone to use. Supported names: all from 'tf.keras.applications'. Please check documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications .
            include_top (bool): If True, the backbone will include it's default top (head).
            **kwargs: The remaining keyward arguments from the backbone. See documentation mentioned above. 
        Returns:
            self.model: The backbone model with specified parameters.
        """

        attributes = getattr(self.applications, backbone_name)
        self.include_top = include_top

        if callable(attributes):
            self.model = attributes(include_top, **kwargs)
            if not include_top:
                return self
            return self.model
        else:
            raise(ValueError(f"The backbone name ({backbone_name}) is not valid. Please check the tf.kera.applications documentation."))

    def add_top(self, n_classes: int, dropout: float, nodes: int):
        """
        Adds the top to model backbone (stored in self.model). It should be chained to 'set_backbone()'.
        The top architecture is:
        (in) -> BatchNormalization -> Dropout -> Dense -> BatchNormalization -> Dropout -> Dense -> (out)
        
        Args:
            n_classes (int): The number of output classes.
            dropout (float): The dropout rate.
            nodex (int): The number of neurond in the dense layer.
        
        Returns:
            self.model: Updates model which already has the backbone.

        """
        if not self.include_top:
            x = tf.keras.layers.BatchNormalization()(self.model.output)
            x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.Dense(nodes, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout)(x)

            output = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)
            self.model = tf.keras.Model(inputs=self.model.inputs, outputs=output)
            
            return self.model
        else:
            raise(ValueError(f"The backbone already has a top. Set 'include_top=False' when calling 'set_backbone', then call 'add_top'."))

