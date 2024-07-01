import tensorflow as tf
import keras
import numpy as np
import warnings

from keras.applications import (
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtXLarge,
)
KERAS_VERSION = keras.__version__
TF_VERSION = tf.__version__
print("tensorflow version", TF_VERSION)
print("keras version", KERAS_VERSION)

def load_pretrained_weight(variant, image_size: tuple):
    assert KERAS_VERSION.startswith("2"), f"keras version {KERAS_VERSION} not supported, only support keras version 2.x"
    convnext_variants_map = {
        "convnext_tiny" : ConvNeXtTiny,
        "convnext_small": ConvNeXtSmall,
        "convnext_base" : ConvNeXtBase,
        "convnext_large": ConvNeXtLarge,
        "convnext_xlarge": ConvNeXtXLarge,
    }
    weights_path = convnext_variants_map[variant](
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    ).get_weight_paths()
    return weights_path

def get_dims_depth(variant):
    convnext_dims_depth_map = {
        "convnext_tiny" : ([96, 192, 384, 768], [3, 3, 9, 3]),
        "convnext_small": ([96, 192, 384, 768], [3, 3, 27, 3]),
        "convnext_base" : ([128, 256, 512, 1024], [3, 3, 27, 3]),
        "convnext_large": ([192, 384, 768, 1536], [3, 3, 27, 3]),
        "convnext_xlarge":([256, 512, 1024, 2048], [3, 3, 27, 3]),
    }
    return convnext_dims_depth_map[variant]

class UserError(Exception):
    pass

@keras.saving.register_keras_serializable("CustomLayers", name="custom_global_pooling")
class GlobalPooling2D(keras.layers.Layer):
    """
    modified from : https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py#L51
    reference : https://arxiv.org/abs/1809.08264
    """
    def __init__(self, pool_func: str, activation='linear', **kwargs):
        """
        custom global pooling layer, also with weight and activation

        :param pool_func: "avg" or "max" for non-weighted pooling, "w_avg" or "w_max" for weighted pooling
        :param activation: can be str or a function that accept 1 argument, i.e, `z`. default `linear`
        """
        super().__init__(**kwargs)
        assert pool_func in ['max', 'avg', 'w_max', 'w_avg'], "one of : 'max', 'avg','w_max', 'w_avg'"
        if isinstance(activation, str):
            self.act = keras.layers.Activation(activation)
        elif callable(activation):
            self.act = activation
        else:
            raise UserError("activation should be either a function or a string")
        self.pool_func = pool_func
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        if self.pool_func.startswith("w_"):
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[1], input_shape[2], 1),
                                          initializer='ones',
                                          trainable=True)
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer='zeros',
                                        trainable=True)
        else:
            pass
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3],

    def call(self, x):
        if "w_avg" == self.pool_func:
            z = tf.reduce_mean(x * self.kernel, axis=(1, 2)) + self.bias
        elif "w_max" == self.pool_func:
            z = tf.reduce_max(x * self.kernel, axis=(1, 2)) + self.bias
        elif "max" == self.pool_func:
            z = tf.reduce_max(x, axis=(1, 2))
        elif "avg" == self.pool_func:
            z = tf.reduce_mean(x, axis=(1, 2))
        else:
            raise UserError("unknown pooling function :", self.pool_func)
        x = self.act(z)
        return x


@keras.saving.register_keras_serializable("CustomLayers", name="stochastic_depth")
class StochasticDepth(keras.layers.Layer):
    """
    source : https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/convnext.py#L140
    """

    def __init__(self, drop_path_rate: float, **kwargs):
        """
        stochastic depth layer

        :param drop_path_rate: float
        """
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


@keras.saving.register_keras_serializable("CustomLayers", name="layer_scale")
class LayerScale(keras.layers.Layer):
    """
    source : https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/convnext.py#L177
    """
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma = None

    def build(self, _):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


""" 
===========================================================================
====== ConvNext components (stage, block, stem, downsampling, etc..) ======
===========================================================================
"""
def convnext_block(x: tf.Tensor, dim: int, stage: int, block:int ,
                   pretrained=None,
                   layer_scale_init_value=1e-6,
                   drop_path_rate=None,
                   variant='convnext_small',
                   name=''
                   ) -> tf.Tensor:
    """
    ConvNext block
    :param x: input tensor
    :param dim: layers dimension
    :param stage: current stage
    :param block: current block
    :param pretrained: pretrained weights, returned by `get_weight_paths` method of keras.Model (keras v2)
    :param layer_scale_init_value: initial value for the layer scale
    :param drop_path_rate: drop path rate for Stochastic Depth
    :param variant: ConvNext variant of the pretrained weights
    :param name: unique identifier for this block
    :return: output tensor
    """
    depthwise_convolution = keras.layers.Conv2D(dim,
                                                kernel_size=7,
                                                padding="same",
                                                groups=dim,
                                                name=f'{name}_{stage}-{block}_depthwise_conv'
                                                )
    layer_normalization = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name}_{stage}-{block}_layernorm')
    pointwise_convolution_1 = keras.layers.Dense(4 * dim, name=f'{name}_{stage}-{block}_pointwise_conv1')
    GELU = keras.layers.Activation("gelu", name=f'{name}_{stage}-{block}_gelu')
    pointwise_convolution_2 = keras.layers.Dense(dim, name=f'{name}_{stage}-{block}_pointwise_conv2')
    add = keras.layers.Add(name=f'{name}_{stage}-{block}_output')
    layer_scale = LayerScale(layer_scale_init_value, projection_dim=dim, name=f'{name}_{stage}-{block}_layer_scale')

    o = depthwise_convolution(x)
    o = layer_normalization(o)
    o = pointwise_convolution_1(o)
    o = GELU(o)
    o = pointwise_convolution_2(o)
    o = layer_scale(o)
    if drop_path_rate is not None:
        o = StochasticDepth(
            drop_path_rate, name=f'{name}_{stage}-{block}_stochastic_depth'
        )(o)
    else:
        o = keras.layers.Activation("linear", name=f'{name}_{stage}-{block}_identity')(o)
    if pretrained:
        try:
            weights = {
                'conv': [
                    pretrained[f'{variant}_stage_{stage}_block_{block}_depthwise_conv.kernel'].numpy(),
                    pretrained[f'{variant}_stage_{stage}_block_{block}_depthwise_conv.bias'].numpy(),
                ],
                'lnorm': [
                    pretrained[f'{variant}_stage_{stage}_block_{block}_layernorm.gamma'].numpy(),
                    pretrained[f'{variant}_stage_{stage}_block_{block}_layernorm.beta'].numpy(),
                ],
                'pointwise1': [
                    pretrained[f'{variant}_stage_{stage}_block_{block}_pointwise_conv_1.kernel'].numpy(),
                    pretrained[f'{variant}_stage_{stage}_block_{block}_pointwise_conv_1.bias'].numpy(),
                ],
                'pointwise2': [
                    pretrained[f'{variant}_stage_{stage}_block_{block}_pointwise_conv_2.kernel'].numpy(),
                    pretrained[f'{variant}_stage_{stage}_block_{block}_pointwise_conv_2.bias'].numpy(),
                ],
                'layer_scale': [pretrained[f'{variant}_stage_{stage}_block_{block}_layer_scale.gamma']]
            }
            depthwise_convolution.set_weights(weights['conv'])
            layer_normalization.set_weights(weights['lnorm'])
            pointwise_convolution_1.set_weights(weights['pointwise1'])
            pointwise_convolution_2.set_weights(weights['pointwise2'])
            layer_scale.set_weights(weights['layer_scale'])
        except KeyError:
            warnings.warn("pretrained weights format invalid, skipping loading weights" +
                          "This must be returned by `get_weight_paths` method of keras.Model (keras v2)",
                          UserWarning)
    return add([o, x])

def patchify_stem(x: tf.Tensor, dim: int, pretrained=None, name='single', variant='convnext_small') -> tf.Tensor:
    """
    patchify stem for ConvNeXt model

    :param x: input tensor
    :param dim: convolutional dimension
    :param pretrained: pretrained weights, returned by `get_weight_paths` method of keras.Model (keras v2)
    :param name: unique identifier for the layers
    :param variant: ConvNext variant of the pretrained weights
    :return: output tensor
    """
    conv = keras.layers.Conv2D(dim, kernel_size=4, strides=4, name=f'{name}_stem_conv')
    layer_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name}_stem_layer_norm')
    o = conv(x)
    o = layer_norm(o)
    if pretrained:
        try:
            conv_w = [
                pretrained[f'{variant}_stem.layer_with_weights-0.kernel'].numpy(),
                pretrained[f'{variant}_stem.layer_with_weights-0.bias'].numpy(),
                ]
            layer_norm_w = [
                pretrained[f'{variant}_stem.layer_with_weights-1.gamma'].numpy(),
                pretrained[f'{variant}_stem.layer_with_weights-1.beta'].numpy(),
            ]
            conv.set_weights(conv_w)
            layer_norm.set_weights(layer_norm_w)
        except KeyError:
            warnings.warn("pretrained weights format invalid, skipping loading weights" +
                          "This must be returned by `get_weight_paths` method of keras.Model (keras v2)",
                          UserWarning)

    return o

def spatial_downsampling(x: tf.Tensor, stage: int, dim: int,
                         pretrained=None,
                         name='single',
                         variant='convnext_small') -> tf.Tensor:
    """
    spatial downsampling layer for ConvNeXt model
    :param x: input tensor
    :param stage: current stage
    :param dim: convolutional dimension
    :param pretrained: pretrained weights, returned by `get_weight_paths` method of keras.Model (keras v2)
    :param name: unique identifier for the layers
    :param variant: ConvNeXt variant of the pretrained weights
    :return: output tensor
    """
    layer_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name}_downsampling_{stage}_layer_norm')
    conv = keras.layers.Conv2D(dim, kernel_size=2, strides=2, name=f'{name}_downsampling_{stage}_conv')
    o = layer_norm(x)
    o = conv(o)
    if pretrained:
        try:
            layer_norm_w = [
                pretrained[f'{variant}_downsampling_block_{stage-1}.layer_with_weights-0.gamma'].numpy(),
                pretrained[f'{variant}_downsampling_block_{stage-1}.layer_with_weights-0.beta'].numpy(),
            ]
            conv_w = [
                pretrained[f'{variant}_downsampling_block_{stage-1}.layer_with_weights-1.kernel'].numpy(),
                pretrained[f'{variant}_downsampling_block_{stage-1}.layer_with_weights-1.bias'].numpy(),
            ]
            conv.set_weights(conv_w)
            layer_norm.set_weights(layer_norm_w)
        except KeyError:
            warnings.warn("pretrained weights format invalid, skipping loading weights" +
                          "This must be returned by `get_weight_paths` method of keras.Model (keras v2)",
                          UserWarning)
    return o

def convnext_stage(x: tf.Tensor, dim: int, depth: int, stage: int,
                   pretrained=None,
                   layer_scale_init_value=1e-6,
                   depth_drop_rates=None,
                   name='stage',
                   variant='convnext_small'
                   ) -> tf.Tensor:
    """
    defining a ConvNeXt stage
    :param x: input tensor
    :param dim: dimension of the convolutional layers
    :param depth: block depth for this stage
    :param stage: current stage
    :param pretrained: pretrained weights, returned by `get_weight_paths` method of keras.Model (keras v2)
    :param layer_scale_init_value: initialization value for layer scale
    :param depth_drop_rates: drop rate for stage depth
    :param name: unique identifier for the layers
    :param variant: ConvNext variant of the pretrained weights
    :return: output tensor
    """
    o = x
    if depth_drop_rates is None:
        depth_drop_rates = np.zeros(depth)
    for j in range(depth):
        o = ConvNext_Block(o,
                           dim=dim,
                           pretrained=pretrained,
                           stage=stage, block=j,
                           layer_scale_init_value=layer_scale_init_value,
                           variant=variant,
                           drop_path_rate=depth_drop_rates[j],
                           name=name)
    return o

def convnext_stage_and_downsampling(x: tf.Tensor, dim: int, depth: int, stage: int,
                                    pretrained,
                                    depth_drop_rates=None,
                                    view='',
                                    variant='convnext_small'
                                    ) -> tf.Tensor:
    """
    defining a ConvNeXt stage and downsampling layers
    :param x: input tensor
    :param dim: convolutional dimension
    :param depth: block depth for this stage
    :param stage: current stage
    :param pretrained: pretrained weights, returned by `get_weight_paths` method of keras.Model (keras v2)
    :param depth_drop_rates: drop rate for stage depth
    :param view: view for multi-view mammography classification model
    :param variant: ConvNext variant of the pretrained weights
    :return: output tensor
    """
    if stage == 0:
        x = keras.layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=f'{variant}_{view}_norm'
        )(x)
        x = patchify_stem(pretrained=pretrained, dim=dim, variant=variant, name=f'{variant}_{view}', x=x)
    else:
        x = spatial_downsampling(pretrained=pretrained,
                                 dim=dim,
                                 stage=stage,
                                 variant=variant,
                                 name=f'{variant}_{view}',
                                 x=x)

    x = ConvNext_Stage(x=x,
                       dim=dim,
                       depth=depth,
                       stage=stage,
                       pretrained=pretrained,
                       depth_drop_rates=depth_drop_rates,
                       name=f'{variant}_{view}_stage',
                       variant=variant)
    return x


def multi_view_fusion_stage(pre_fusion_x: dict, dim: int, depth: int, stage: int,
                            pretrained_weights=None,
                            depth_drop_rates=0.0,
                            fusion_block_index=0,
                            model_var='convnext_small',
                            ) -> tf.Tensor:
    for view, x in pre_fusion_x.items():
        x = spatial_downsampling(x, stage, dim,
                                 pretrained=pretrained_weights,
                                 variant=model_var,
                                 name=f"{model_var}_{view}_fusion_downsampling",
                                 )
        pre_fusion_x[view] = x
    x_dual_skip = pre_fusion_x.copy()
    if depth_drop_rates == 0.0:
        depth_drop_rates = np.zeros(depths[i])
    if fusion_block_index > depth:
        fusion_block_index = depth-1
    # stage iteration here
    for j in range(depth):
        if j < fusion_block_index:
            for view, x_view in pre_fusion_x.items():
                x_view = convnext_block(x_view, dim, stage, block=j,
                                        pretrained=pretrained_weights,
                                        drop_path_rate=depth_drop_rates[j],
                                        variant=model_var,
                                        name=f'{model_var}_{view}_fusion_stage')
                pre_fusion_x[view] = x_dual_skip[view] = x_view
            continue
        elif j == fusion_block_index:
            x = keras.layers.Average(name=f'{model_var}_fusion_merge')(list(pre_fusion_x.values()))
            x = convnext_block(x, dim, stage, block=j,
                               pretrained=pretrained_weights,
                               drop_path_rate=depth_drop_rates[j],
                               variant=model_var,
                               name=f'{model_var}_post-fusion_stage')
            x = keras.layers.Add(name="merge_fused_and_examined_skip")([x, x_dual_skip["Examined"]])
            continue
        elif j > fusion_block_index:
            x = convnext_block(x, dim, stage, block=j,
                               pretrained=pretrained_weights,
                               drop_path_rate=depth_drop_rates[j],
                               variant=model_var,
                               name=f'{model_var}_post-fusion_stage')
    return x


if __name__ == "__main__":
    X = np.random.uniform(size=(100, 32, 32, 3))
    y = np.random.uniform(size=(100, 1))
