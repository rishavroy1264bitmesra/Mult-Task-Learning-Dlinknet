import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, \
    Add, Activation, \
    BatchNormalization
from tensorflow.python.keras.models import Model

# ## Constants

# In[2]:

OUTPUT_SHAPE = (1024, 1024, 1)
INPUT_SHAPE = (1024, 1024, 3)
BATCH_SIZE = 5
EPOCHS = 30
PARALLEL_THREADS_TF_MAPPING = 4
IMAGES_PATH = "/home/ubuntu/valdata/lb_dataset/images"
MODEL_PATH = "/home/ubuntu/bigvolume/lb_training/lb_models"
TENSORBOARD_PATH = "/home/ubuntu/bigvolume/lb_training/lb_tensorboard"


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def log_cosh_dce_loss(y_true, y_pred):
    """
    Implementation suggested in https://arxiv.org/pdf/2006.14822.pdf
    """
    return tf.math.log(tf.math.cosh(dice_loss(y_true, y_pred)))


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + log_cosh_dce_loss(y_true, y_pred)
    return loss


def residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    input_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    res_tensor = Add()([input_tensor, x])
    res_tensor = Activation('relu')(res_tensor)
    return res_tensor


def dilated_center_block(input_tensor, num_filters):
    """
    :param input_tensor:
    :param num_filters:
    :return:
    """
    dilation_1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(input_tensor)
    dilation_1 = Activation('relu')(dilation_1)

    dilation_2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(dilation_1)
    dilation_2 = Activation('relu')(dilation_2)

    dilation_4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(dilation_2)
    dilation_4 = Activation('relu')(dilation_4)

    dilation_8 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(dilation_4)
    dilation_8 = Activation('relu')(dilation_8)

    final_diliation = Add()([input_tensor, dilation_1, dilation_2, dilation_4, dilation_8])

    return final_diliation


def decoder_block(input_tensor, num_filters):
    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)
    return decoder_tensor


def encoder_block(input_tensor, num_filters, num_res_blocks):
    encoded = residual_block(input_tensor, num_filters)
    while num_res_blocks > 1:
        encoded = residual_block(encoded, num_filters)
        num_res_blocks -= 1
    encoded_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoded)
    return encoded, encoded_pool


def create_mtl_dlinknet():
    """
    Implementation of Multi-Task Learning model using D-Linknet Backend
    :return:
    """
    inputs = Input(shape=INPUT_SHAPE)
    inputs_ = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    inputs_ = BatchNormalization()(inputs_)
    inputs_ = Activation('relu')(inputs_)
    max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)

    # Create Encoders
    encoded_1, encoded_pool_1 = encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
    encoded_2, encoded_pool_2 = encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
    encoded_3, encoded_pool_3 = encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
    encoded_4, encoded_pool_4 = encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)

    # Create center Dilated Block
    center_lane_boundary = dilated_center_block(encoded_4, 512)
    center_lane_marking = dilated_center_block(encoded_4, 512)

    # Create Lane Boundary Task Prediction Head
    lane_boundary_decoded_1 = Add()([decoder_block(center_lane_boundary, 256), encoded_3])
    lane_boundary_decoded_2 = Add()([decoder_block(lane_boundary_decoded_1, 128), encoded_2])
    lane_boundary_decoded_3 = Add()([decoder_block(lane_boundary_decoded_2, 64), encoded_1])
    lane_boundary_decoded_4 = decoder_block(lane_boundary_decoded_3, 64)
    lane_boundary_upsample = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(lane_boundary_decoded_4)
    lane_boundary_output = Conv2D(1, (1, 1), activation='sigmoid', name="lane_boundary_output")(lane_boundary_upsample)

    # Create Lane Marking Task Prediction Head
    lane_marking_decoded_1 = Add()([decoder_block(center_lane_marking, 256), encoded_3])
    lane_marking_decoded_2 = Add()([decoder_block(lane_marking_decoded_1, 128), encoded_2])
    lane_marking_decoded_3 = Add()([decoder_block(lane_marking_decoded_2, 64), encoded_1])
    lane_marking_decoded_4 = decoder_block(lane_marking_decoded_3, 64)
    lane_marking_upsample = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(lane_marking_decoded_4)
    lane_marking_output = Conv2D(1, (1, 1), activation='sigmoid', name="lane_marking_output")(lane_marking_upsample)

    model_i = Model(inputs=[inputs], outputs=[lane_boundary_output, lane_marking_output],
                    name="multi_task_d-linknet_lb_lm_model")

    mtl_loss = {
        'lane_boundary_output': bce_dice_loss,
        'lane_marking_output': bce_dice_loss}

    mtl_loss_weights = {
        'lane_boundary_output': 0.8,
        'lane_marking_output': 0.2}
    mtl_metrics = {
        'lane_boundary_output': dice_coeff,
        'lane_marking_output': dice_coeff}

    model_i.compile(optimizer='adam', loss=mtl_loss, metrics=mtl_metrics, loss_weights=mtl_loss_weights)
    model_i.summary()
    # model_i.load_weights(save_model_path)
    return model_i


# In[6]:


model = create_mtl_dlinknet()
