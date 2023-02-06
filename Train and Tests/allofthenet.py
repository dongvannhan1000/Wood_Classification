import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.nn import local_response_normalization
from keras.models import Model
from keras import layers
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU,PReLU, Activation, Concatenate, Lambda, GlobalAveragePooling2D, Dropout,BatchNormalization,Lambda
from keras.optimizers import Adam , RMSprop
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.models import Model

from functools import partial

import warnings



#Note: All CNNs had been reduced because it consumes lots of time to train in my shitty computer.
#      Please notice that all of those CNNs is suitable for small applications as Wood Identification.
#      You guys are welcome to change them. (Trinh Duc Tinh - K18)


'''
Preparation module.
'''



def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              name = None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 3
    x = layers.Conv2D(
        filters, kernel_size,
        strides=(strides,strides),
        padding=padding,
        use_bias=False)(x)    
    '''
    x =   tfa.layers.GroupNormalization(
                                    groups = 1,
                                    axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)
    '''
    x =   tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(x)

    if(activation == 'relu'): 
        x = LeakyReLU(alpha = 0.1)(x)
    return x












def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='Block35'`
        - Inception-ResNet-B: `block_type='Block17'`
        - Inception-ResNet-C: `block_type='Block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals before adding
            them to the shortcut branch.
        block_type: `'Block35'`, `'Block17'` or `'Block8'`, determines
            the network structure in the residual branch.
        block_idx: used for generating layer names.
        activation: name of the activation function to use at the end
            of the block (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'Block35'`,
            `'Block17'` or `'Block8'`.
    """
    channel_axis =  3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)
    scale = 8
    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32/2, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32/2, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32/2, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        #branch_2 = conv2d_bn(branch_2, 48/2, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 64/2, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 192/scale, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128/scale, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 160/scale, [1, 3], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 192/scale, [3, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192/scale, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192/scale, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 224/scale, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 256/scale, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Block35", "Block17" or "Block8", '
                         'but got: ' + str(block_type))

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   name=name_fmt('Conv2d_1x1'))
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name_fmt('ScaleSum'))([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x









def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False,LRN = True):
    
    shortcut = y

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', use_bias=False)(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(y)
    if LRN:
        y = tf.keras.layers.Lambda(tf.nn.local_response_normalization)(y)

    if _project_shortcut or _strides != (1, 1):
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', use_bias=False)(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y



'''
End of preparation.
'''









'''
AlexNet.
'''
def myalexnet(cfg,verbose = 1):
    input_image = Input(shape=(cfg['image_size'],cfg['image_size'],cfg['channel_no']))
    model = Conv2D(filters=64, kernel_size=(7,7), strides=(4,4), activation='relu', padding="same")(input_image)
    model = Lambda(local_response_normalization)(model)
    model = MaxPool2D(pool_size=(3,3), strides=(1,1))(model)
    model = Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(model)
    model = Lambda(local_response_normalization)(model)
    model = MaxPool2D(pool_size=(3,3), strides=(1,1))(model)
    model = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = Lambda(local_response_normalization)(model)
    model = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = Lambda(local_response_normalization)(model)
    model = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = Lambda(local_response_normalization)(model)
    model = MaxPool2D(pool_size=(3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(2048, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(2048, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(cfg['class_no'], activation='softmax')(model)
    model = Model(inputs=input_image, outputs=model)
    model.compile(optimizer=Adam(lr = 0.001, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    if(verbose): 
        model.summary()
    return model



'''
VGG.
'''


def vgg(cfg):
    input_image = Input(shape=(cfg['image_size'],cfg['image_size'],cfg['channel_no']))
    scale = 4
    model = conv2d_bn(input_image,64/scale, (3,3), activation='relu')
    model = conv2d_bn(model,64/scale, (3,3), activation='relu')
    model = MaxPool2D()(model)

    model = conv2d_bn(model,128/scale, (3,3), activation='relu')
    model = conv2d_bn(model,128/scale, (3,3), activation='relu')
    model = MaxPool2D()(model)


    scale = 8
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = MaxPool2D()(model)

    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = conv2d_bn(model,512/scale, (3,3), activation='relu')
    model = MaxPool2D()(model)


    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(cfg['class_no'], activation='softmax')(model)
    model = Model(input_image, model, name='vgg19')
    model.compile(optimizer=Adam(lr = 0.01, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model









'''
Inception_ResNet_v2
'''

def InceptionResNetV2(cfg,verbose = 1):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    channel_axis = 3
    scale = 4
    input_image = tf.keras.layers.Input(shape=(cfg['image_size'],cfg['image_size'],cfg['channel_no']))

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(input_image, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 64, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')

    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 64, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 64, 3, padding='valid', name='Conv2d_4a_3x3')
    #x = MaxPooling2D(3, strides=2, name='MaxPool_5a_3x3')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    channel_axis =  3
    name_fmt = partial(_generate_layer_name, prefix='Mixed_5b')
    branch_0 = conv2d_bn(x, 96, 1, name=name_fmt('Conv2d_1x1', 0))
    branch_1 = conv2d_bn(x, 48, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 64, 5, name=name_fmt('Conv2d_0b_5x5', 1))
    branch_2 = conv2d_bn(x, 64, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0c_3x3', 2))
    branch_pool = AveragePooling2D(3,
                                   strides=1,
                                   padding='same',
                                   name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, name=name_fmt('Conv2d_0b_1x1', 3))
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_5b')(branches)

    # 10x Block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1,10):
        x = _inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='Block35',
                                    block_idx=block_idx)


    scale = 4
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x,
                         384/scale,
                         3,
                         strides=1,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256/scale, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    #branch_1 = conv2d_bn(branch_1, 256/scale, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1,
                         384/scale,
                         3,
                         strides=1,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3,
                               strides=1,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    '''
    # 20x Block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 2):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256/scale, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0,
                         384/scale,
                         3,
                         strides=1,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1,
                         288/scale,
                         3,
                         strides=1,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256/scale, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    #branch_2 = conv2d_bn(branch_2, 288/scale, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2,
                         320/(scale),
                         3,
                         strides=1,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 10x Block8 (Inception-ResNet-C block): 8 x 8 x 2080

    '''    
    for block_idx in range(1, 10):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,
                                scale=1.,
                                activation=None,
                                block_type='Block8',
                                block_idx=10)

    # Final convolution block
    x = conv2d_bn(x, 256, 1, name='Conv2d_7b_1x1')

    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - 0.5, name='Dropout')(x)
    x = Dense(cfg['class_no'], name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)

    # Create model
    model = Model(input_image, x, name='inception_resnet_v2')
    model.compile(optimizer=Adam(lr = 0.01, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    if(verbose):
        model.summary()

    return model



'''
ResNet and ResNet_LRN.
''' 
def ResNet(cfg,LRN=False,verbose = 1):
    
    inputs = Input(shape=(cfg['image_size'],cfg['image_size'],cfg['channel_no']))
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    res1 = residual_block(act1, 64,LRN=LRN);  
    pool1 = MaxPool2D(pool_size=(2,2))(res1)
    
    res2 = residual_block(pool1, 64,LRN=LRN);  
    pool2 = MaxPool2D(pool_size=(2,2))(res2)
        
    res3 = residual_block(pool2, 64,LRN=LRN)
    pool3 = MaxPool2D(pool_size=(2,2))(res3)
    
    res4 = residual_block(pool3, 64,LRN=LRN)
    pool4 = MaxPool2D(pool_size=(2,2))(res4)
    
    flat1 = Flatten()(pool4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(cfg['class_no'], activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(lr = 0.001, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()
    
    return model




def ResNet_LRN(cfg,verbose = 1):

    model = ResNet(cfg,LRN = True,verbose = verbose)

    return model



MyNetList = {
            "alexnet":              myalexnet,
            "vgg" :                 vgg,
            "Inception_ResNet_V2" : InceptionResNetV2,
            "ResNet":               ResNet,
            "ResNet_LRN":           ResNet_LRN,
            }


