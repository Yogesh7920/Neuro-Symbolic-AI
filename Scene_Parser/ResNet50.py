import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


class ResNet:
    def __init__(self):
        feature_map_model = tf.keras.applications.ResNet50(include_top=True, weights=None)
        self.model = feature_map_model
        self.C5 = feature_map_model.layers[-3].output
        self.C4 = feature_map_model.layers[-35].output
        self.C3 = feature_map_model.layers[-97].output
        self.C2 = feature_map_model.layers[-139].output





# class BatchNorm(KL.BatchNormalization):
#     """Extends the Keras BatchNormalization class to allow a central place
#     to make changes if needed.
#     Batch normalization has a negative effect on training if batches are small
#     so this layer is often frozen (via setting in Config class) and functions
#     as linear layer.
#     """
#     def call(self, inputs, training=None):
#         """
#         Note about training values:
#             None: Train BN layers. This is the normal mode
#             False: Freeze BN layers. Good when batch size is small
#             True: (don't use). Set layer in training mode even when making inferences
#         """
#         return super(self.__class__, self).call(inputs, training=training)


# def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
#     """Build a ResNet graph.
#         architecture: Can be resnet50 or resnet101
#         stage5: Boolean. If False, stage5 of the network is not created
#         train_bn: Boolean. Train or freeze Batch Norm layers
#     """
#     assert architecture in ["resnet50", "resnet101"]
#     # Stage 1
#     x = KL.ZeroPadding2D((3, 3))(input_image)
#     x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
#     x = BatchNorm(name='bn_conv1')(x, training=train_bn)
#     x = KL.Activation('relu')(x)
#     C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
#     # Stage 2
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
#     C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
#     # Stage 3
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
#     C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
#     # Stage 4
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
#     block_count = {"resnet50": 5, "resnet101": 22}[architecture]
#     for i in range(block_count):
#         x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
#     C4 = x
#     # Stage 5
#     if stage5:
#         x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
#         x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
#         C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
#     else:
#         C5 = None
#     return [C1, C2, C3, C4, C5]