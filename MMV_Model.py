import keras
import tensorflow as tf
import numpy as np

class D:
    # param for dataset
    image_size = (512, 288)
    splits = ['train', 'valid']
    views = ['Examined', 'Aux']

    # param for ConvNeXt multi-view model
    model_var = 'convnext_small'
    dropout_rate = 0.2
    pooling = 'avg'
    fusion_stage = 3
    fusion_index = 0
    fc_layers_depth = 1
    fc_layers_dims = 512

    # param for training
    epochs = 150
    loss_fn = 'categorical_crossentropy'
    optimizer = lambda: keras.optimizers.Adam(1e-5)
    metrics = lambda: [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.F1Score(average='macro', name='macro_F1')
    ]

def get_dims_depth(variant=D.model_var):
    convnext_dims_depth_map = {
        "convnext_tiny": ([96, 192, 384, 768], [3, 3, 9, 3]),
        "convnext_small": ([96, 192, 384, 768], [3, 3, 27, 3]),
        "convnext_base": ([128, 256, 512, 1024], [3, 3, 27, 3]),
        "convnext_large": ([192, 384, 768, 1536], [3, 3, 27, 3]),
        "convnext_xlarge": ([256, 512, 1024, 2048], [3, 3, 27, 3]),
    }
    return convnext_dims_depth_map[variant]

@keras.saving.register_keras_serializable()
class GlobalPooling2D(keras.layers.Layer):
    '''
    modified from : https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py#L51
    reference : https://arxiv.org/abs/1809.08264
    '''
    def __init__(self,pool_func:str, activation:str='linear', **kwargs):
        super().__init__(**kwargs)
        assert pool_func in ['max', 'avg','w_max', 'w_avg'], "one of : 'max', 'avg','w_max', 'w_avg'"
        self.act = activation
        self.pool_func = pool_func

    def build(self, input_shape):
        if "w_" in self.pool_func:
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
            z = tf.reduce_mean(x*self.kernel, axis=(1, 2)) + self.bias
        elif "w_max" == self.pool_func:
            z = tf.reduce_max(x*self.kernel, axis=(1, 2)) + self.bias
        elif "max" == self.pool_func:
            z = tf.reduce_max(x, axis=(1, 2))
        elif "avg" == self.pool_func:
            z = tf.reduce_mean(x, axis=(1, 2))
        x = keras.layers.Activation(self.act)(z)
        return x

@keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """
    source : https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/convnext.py#L140
    """
    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            random_tensor = tf.cast(random_tensor, TOUT)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config

@keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """
    source : https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/convnext.py#L177
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

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


def ConvNext_Block(x, dim,
                   stage=0,
                   block=0,
                   layer_scale_init_value=1e-6,
                   drop_path_rate=None,
                   name=''
                   ):
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

    return add([o, x])


def patchify_stem(x,
                  dim=96,
                  name='single',
                  ):
    conv = keras.layers.Conv2D(dim,
                               kernel_size=4,
                               strides=4,
                               name=f'{name}_stem_conv')
    layer_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name}_stem_layer_norm')
    o = conv(x)
    o = layer_norm(o)

    return o


def spatial_downsampling(x,
                         stage,
                         dim,
                         kernel_size=2,
                         stride=2,
                         name='single',
                         ):
    layer_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name}_downsampling_{stage}_layer_norm')
    conv = keras.layers.Conv2D(dim,
                               kernel_size=kernel_size,
                               strides=stride,
                               name=f'{name}_downsampling_{stage}_conv'
                               )
    o = layer_norm(x)
    o = conv(o)

    return o


def ConvNext_Stage(x,
                   dim,
                   depth,
                   stage,
                   layer_scale_init_value=1e-6,
                   drop_path_rate=0,
                   name='stage',
                   ):
    o = x
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, depth)
    ]
    for j in range(depth):
        o = ConvNext_Block(o,
                           dim=dim,
                           stage=stage, block=j,
                           layer_scale_init_value=layer_scale_init_value,
                           drop_path_rate=depth_drop_rates[j],
                           name=name
                           )
    return o


def ConvNext_stage_and_downsampling(x,
                                    dim,
                                    depth,
                                    i,
                                    drop_path_rate=0,
                                    view='single',
                                    variant='convnext_small'
                                    ):
    if i == 0:
        x = keras.layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=f'{variant}_{view}_norm'
        )(x)
        x = patchify_stem(dim=dim,
                          name=f'{variant}_{view}', x=x)
    else:
        x = spatial_downsampling(dim=dim,
                                 stage=i,
                                 kernel_size=2,
                                 stride=2,
                                 name=f'{variant}_{view}', x=x)

    x = ConvNext_Stage(x=x,
                       dim=dim,
                       depth=depth,
                       stage=i,
                       drop_path_rate=drop_path_rate,
                       name=f'{variant}_{view}_stage',
                       )
    return x


def multi_view_fusion_stage(pre_fusion,
                            i,
                            dims=None,
                            depths=None,
                            drop_path_rate=0,
                            fusion_block_index=0,
                            model_var=D.model_var
                            ):
    for view, x in pre_fusion.items():
        x = spatial_downsampling(
                                 dim=dims[i],
                                 stage=i,
                                 kernel_size=2,
                                 stride=2,
                                 name=f"{model_var}_{view}_fusion_downsampling",
                                 x=x
                                 )
        pre_fusion[view] = x
    x_dual_skip = pre_fusion.copy()
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, depths[i])
    ]
    # stage iteration here
    for j in range(depths[i]):
        if j < fusion_block_index:
            for view, x in pre_fusion.items():
                x = ConvNext_Block(x,
                                   dims[i], stage=i,
                                   block=j,
                                   drop_path_rate=depth_drop_rates[j],
                                   name=f'{model_var}_{view}_fusion_stage')
                pre_fusion[view] = x_dual_skip[view] = x
            continue
        elif j == fusion_block_index:
            x = keras.layers.Average(name=f'{model_var}_fusion_merge')(list(pre_fusion.values()))
            x = ConvNext_Block(x,
                               dims[i], stage=i,
                               block=j,
                               drop_path_rate=depth_drop_rates[j],
                               name=f'{model_var}_post-fusion_stage')
            x = keras.layers.Add(name="merge_fused_and_examined_skip")([x, x_dual_skip["Examined"]])
            continue
        elif j > fusion_block_index:
            x = ConvNext_Block(x,
                               dims[i], stage=i,
                               block=j,
                               drop_path_rate=depth_drop_rates[j],
                               name=f'{model_var}_post-fusion_stage')
    return x

def get_inputs():
    inputs = {
                "Examined" : keras.Input(shape=[*D.image_size,3], name='Examined', dtype=tf.float32),
                "Aux" : keras.Input(shape=[*D.image_size,3], name='Aux', dtype=tf.float32)
             }
    return inputs

def model_compile(model):
    model.compile(
        loss=D.loss_fn,
        optimizer=D.optimizer(),
        metrics=D.metrics(),
    )
    return model


def create_model(model_var=D.model_var,
                 fusion_stage=D.fusion_stage,
                 fusion_block_index=D.fusion_index,
                 fc_layers_depth=D.fc_layers_depth,
                 fc_layers_dims=D.fc_layers_dims,
                 drop_path_rate=D.dropout_rate,
                 pooling=D.pooling,
                 ):
    inputs = get_inputs()
    pre_fusion = {key: value for key, value in inputs.items()}
    dims, depths = get_dims_depth(model_var)
    for i in range(len(dims)):
        if i < fusion_stage:
            for key in D.views:
                pre_fusion[key] = ConvNext_stage_and_downsampling(pre_fusion[key],
                                                                  dims[i],
                                                                  depths[i], i,
                                                                  variant=model_var,
                                                                  drop_path_rate=drop_path_rate,
                                                                  view=key)
            continue
        if i == fusion_stage:
            x = multi_view_fusion_stage(pre_fusion,
                                        i,
                                        dims=dims,
                                        depths=depths,
                                        drop_path_rate=drop_path_rate,
                                        fusion_block_index=fusion_block_index,
                                        )
            continue
        x = ConvNext_stage_and_downsampling(x, dims[i], depths[i], i,
                                            view="fine",
                                            drop_path_rate=drop_path_rate,
                                            variant=model_var)
    x = GlobalPooling2D(pooling, name=f'{model_var}_global_pooling')(x)
    LN1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{model_var}_pre_FC_ln')
    x = LN1(x)
    for i in range(fc_layers_depth):
        x = keras.layers.Dense(fc_layers_dims, activation='gelu', name=f'{model_var}_cls_{i}')(x)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(5, activation='softmax', dtype='float32', name=f'{model_var}_output')(x)
    model = model_compile(keras.Model(inputs, output, name=f'{model_var}_mammo_multi_view'))
    return model

