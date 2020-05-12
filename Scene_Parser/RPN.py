import tensorflow as tf


class RPN(tf.keras.Model):

    #     anchor_stride: if value is n then 1 in n pixel chosen from feature map as anchor
    #     anchor_boxes: number of possible types of boxes for a given anchor

    def __init__(self, ResNet_size, anchor_stride, anchor_boxes):
        super(RPN, self).__init__()

        self.Shared = tf.keras.layers.Conv2D(ResNet_size, kernel_size=3, activation='relu',
                                             padding='same', strides=anchor_stride)
        self.Box_reg = tf.keras.layers.Conv2D(4 * anchor_boxes, kernel_size=1, activation='linear',
                                              padding='valid')
        self.Box_class = tf.keras.layers.Conv2D(2 * anchor_boxes, kernel_size=1, activation='linear',
                                                padding='valid')

        #     TODO: Try changing to Fully-connected layers

    def call(self, feature_map):
        x = self.Shared(feature_map)

        box_delta = self.Box_reg(x)
        box_delta = tf.reshape(box_delta, (box_delta.shape[0], -1, 4))

        box_logits = self.Box_class(x)
        box_logits = tf.reshape(box_logits, (box_logits.shape[0], -1, 2))

        box_probs = tf.nn.softmax(box_logits, axis=-1)

        return box_delta, box_logits, box_probs

