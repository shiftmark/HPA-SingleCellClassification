from tensorflow.keras.layers import Activation, Conv2D, concatenate, Conv2DTranspose, Dropout, Input, MaxPool2D
from tensorflow.keras import Model

class Unet():

    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
    
    @staticmethod
    def conv2d_block(input_tensor, n_filters, kernel_size, level, side):
        '''
        Adds 2 conv layers. 
        
        Args: 
            input_tensor (tensor) - the input tensor;
            n_filters (int) - the number of convolution filters;
            kernel_size (int) - the kernel size for the convolution;
            level (string) - on which level the layer is;
            side (string) - Encoder, Bottleneck or Decoder.
                
        Returns:
            tensor of output features.        
        '''

        x = input_tensor
        for i in range(2):
            x = Conv2D(n_filters, kernel_size, kernel_initializer='he_normal', padding='same', name=f'Conv-{i}_{level}_{side}')(x)
            x = Activation('relu', name=f'Activ-{i}_{level}_{side}')(x)
        
        return x

    @classmethod
    def encoder_block(cls, input_tensor, level, n_filters=64, kernel_size=(3, 3), pool_size=(2, 2), dropout=0.3):
        '''
        Adds two conv blocks and then performs down sampling on output of convolutions.
        
        Args:
            input_tensor (tensor) - the input tensor;
            n_filters (int) - the number of filters;
            pool_size (tuple of 2 ints) - the pooling window size, for MaxPooling layers;
            dropout (float) - the neuron dropout rate for Dropout layers;
            level (string) - on which level the layer is.
            
            
        Returns:
            features (tensor) - the output features of the conv block;
            down_step (tensor) - the max pooled features with dropout.
        '''
        
        features = cls.conv2d_block(input_tensor=input_tensor, n_filters=n_filters, kernel_size=kernel_size, side='Encoder', level=level)
        down_step = MaxPool2D(pool_size=pool_size, name=f'MaxP-{level}-Encoder')(features)
        down_step = Dropout(rate=dropout, name=f'Drop-{level}-Encoder')(down_step)
                    
        return features, down_step

    @classmethod
    def encoder(cls, inputs):
        '''
        Defines the encoder (downsampling path).
        
        Args:
            inputs (tensor) - batch of input images.
            
        Returns:
            (features1, features2, features3, features4) (tuple of 4 tensors) - the output features or all encoder blocks;
            down_step4 (tensor) - the output (maxpooled) features of the last encoder block.
            
        '''
        
        features1, down_step1 = cls.encoder_block(inputs, level='Level_1')
        features2, down_step2 = cls.encoder_block(down_step1, n_filters=128, level='Level_2')
        features3, down_step3 = cls.encoder_block(down_step2, n_filters=256, level='Level_3')
        features4, down_step4 = cls.encoder_block(down_step3, n_filters=512, level='Level_4')
        
        return down_step4, (features1, features2, features3, features4)

    @classmethod
    def bottleneck(cls, inputs):
        '''
        Defines the bottleneck convolutions, to extact more features before upsampling.
        
        Args: 
            inputs (tensor) - input tensor;
            bottle_neck (tensor) - output of conv block.
        '''
        
        bottle_neck = cls.conv2d_block(inputs, n_filters=1024, kernel_size=(3, 3), level='Level_5', side='Bottleneck')
        
        return bottle_neck

    @classmethod
    def decoder_block(cls, inputs, features, level, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3):
        '''
        Transposes the inputs (from bottleneck) and concatenates features (from decoder).
        
        Args:
            inputs (tensor) - the batch of inputs;
            features (tensor) - the features from the necoder block;
            level (string) - on which level the layer is;
            n_filters (int) - the number of filters;
            kernel_size (tuple of ints) - the kernel size;
            strides (tuple of ints) - the strides of the upsampling layers;
            dropout (float) - the dropout rate for Dropout layer.
            
        Returns: 
        decoder_block_out (tensor) - the output features of the decoder block.
        '''
        
        up_step = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', name=f'Transpose-{level}-Decoder')(inputs)
        concat = concatenate([up_step, features], name=f'Concat-{level}-Decoder')
        concat = Dropout(rate=dropout, name=f'Drop-{level}-Decoder')(concat)
        decoder_block_out = cls.conv2d_block(concat, n_filters=n_filters, kernel_size=kernel_size, level=level, side='Decoder')
        
        return decoder_block_out

    def decoder(self, inputs, features):
        '''
        Defines the decoder (upsampling path).
        
        Args:
            inputs (tensor) - batch of inputs;
            features (tuple) - features for the encoder blocks.
            
        Returns:
            outputs (tensor) - the pixel wise label map of the image.
        '''
        features1, features2, features3, features4 = features
        
        up_step1 = self.decoder_block(inputs, features4, n_filters=512, level='Level_4')
        up_step2 = self.decoder_block(up_step1, features3, n_filters=256, level='Level_3')
        up_step3 = self.decoder_block(up_step2, features2, n_filters=128, level='Level_2')
        up_step4 = self.decoder_block(up_step3, features1, level='Level_1')
        
        outputs = Conv2D(self.n_classes, kernel_size=(1, 1), activation='softmax', name='output')(up_step4)
        
        return outputs

    def unet(self):
        '''
        Define the U-Net model by connecting the encoder, bottleneck and decoder.
        
        Args: -
        Returns:
            model (model) - the U-net model.
        '''
        inputs = Input(shape=self.input_shape, name='input')
        down_step4, features = self.encoder(inputs)
        
        bottle_neck = self.bottleneck(down_step4)
        
        outputs = self.decoder(bottle_neck, features)
        
        model = Model(inputs=inputs, outputs=outputs, name='U-net')
        
        return model