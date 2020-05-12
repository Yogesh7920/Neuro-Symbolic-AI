import tensorflow as tf


class ResNet:
    def __init__(self):
        feature_map_model = tf.keras.applications.ResNet50(include_top=True, weights=None)
        self.model = feature_map_model
        self.C5 = feature_map_model.layers[-3].output
        self.C4 = feature_map_model.layers[-35].output
        self.C3 = feature_map_model.layers[-97].output
        self.C2 = feature_map_model.layers[-139].output

