"""
ResNet-50 implementation
Using pretrained weights from site
"""

from keras.layers import ZeroPadding2D, Conv2D, Activation, Add, Input, Flatten, Dense, MaxPool2D, AveragePooling2D, TimeDistributed
from keras import backend as K
from frcnn.roi_pooling import RoiPoolingConv
from frcnn.batch_norm import FixedBatchNorm

weights_file = "models/resnet50_weights_tf_dim_ordering_tf_kernels.h5"

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_padding:
        input_length += 6
        # 4 convolutions filter sizes
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride

        return input_length
    return get_output_length(width), get_output_length(height)


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=False):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3
    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base+'2a', trainable=trainable)(input_tensor)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b', trainable=trainable)(x)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base+'2c', trainable=trainable)(x)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=False):
    nb_filter1, nb_filter2, nb_filter3=  filters
    bn_axis = 3
    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base+'2a', trainable=trainable)(input_tensor)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b', trainable=trainable)(x)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base+'2c', trainable=trainable)(x)
    x = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base+'1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNorm(axis=bn_axis, name=bn_name_base+'1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def nn_base(input_tensor=None, trainable=False):
    input_shape = (None, None, 3)
    bn_axis = 3
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = ZeroPadding2D((3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNorm(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

    return x


def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D(pool_size=(7, 7)), name='avg_pool')(x)

    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name="rpn_out_classifier")(x)
    x_reger = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name="rpn_out_regressor")(x)

    return [x_class, x_reger, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, pooling_regions, pooling_regions, 1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='rcnn_classifier_{}'.format(nb_classes))(out)
    out_reger = TimeDistributed(Dense((nb_classes - 1) * 4 , activation='linear', kernel_initializer='zero'), name='rcnn_regressor_{}'.format(nb_classes))(out)

    return [out_class, out_reger]


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNorm(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x
