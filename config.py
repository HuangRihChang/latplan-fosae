import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import keras.backend as K
# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))

def load_session(GPU=True):
    if GPU:
        K.set_session(
            tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    allow_soft_placement=True,
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1,
                    device_count = {'CPU': 1, 'GPU': 1},
                    gpu_options =
                    tf.compat.v1.GPUOptions(
                        per_process_gpu_memory_fraction=1.0,
                        allow_growth=True,))))
    else:
        K.set_session(
            tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1,
                    device_count = {'CPU': 1, 'GPU': 0},
                    gpu_options =
                    tf.GPUOptions(
                        per_process_gpu_memory_fraction=1.0,
                        allow_growth=True,))))


if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('Using GPU')
    GPU=True
else:
    print('Using CPU only')
    GPU=False

load_session(GPU)
clear_session = K.clear_session
print(globals())

def reload_session(GPU=GPU):
    clear_session()
    load_session()
