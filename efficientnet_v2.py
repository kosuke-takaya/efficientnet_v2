import os
import shutil
import yaml
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras_efficientnet_v2
from attrdict import AttrDict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def predict(cfg):
    # Initial settings
    datagen_kwargs = dict(rescale=1. / 255)
    dataflow_kwargs = dict(target_size=(cfg.img_size, cfg.img_size),
                            batch_size=cfg.batch_size,
                            interpolation="bilinear")

    # Load test images
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    test_generator = test_datagen.flow_from_directory(
        cfg.test_data_dir, shuffle=False, **dataflow_kwargs)

    # Correct labels
    y_true = test_generator.classes

    print(len(test_generator._filepaths))

    # Load trained model
    model = set_network(cfg, test_generator.num_classes)
    model.load_weights(cfg.output_model)
    # model.summary()

    # Predict labels
    predicts = model.predict(test_generator)
    y_pred = np.argmax(predicts, axis=1)

    # Accuracy calculation
    acc = np.sum(y_pred == y_true)/len(y_pred)*100
    cm = confusion_matrix(y_pred, y_true)
    cm = pd.DataFrame(data=cm, index=test_generator.class_indices,
                           columns=test_generator.class_indices)
    print(f'acc: {acc:.2f}%')
    print(f'confusion_matrix:\n {cm}')

    print(test_generator.class_indices)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.savefig(os.path.join(os.path.dirname(cfg.output_model), 'confusion_matrix.png'))
    print('Saved confusion_matrix')
    print(os.path.join(os.path.dirname(cfg.output_model), 'confusion_matrix.png'))
    plt.close()

    # Misclassification
    mispath = os.path.join(os.path.dirname(cfg.output_model), "misclassification")
    if os.path.exists(mispath):
        shutil.rmtree(mispath)
    os.makedirs(mispath, exist_ok=True)
    for f in list(np.asarray(test_generator._filepaths)[y_pred != y_true]):
        file_to = os.path.join(mispath, *f.split("/")[-2:])
        os.makedirs(os.path.dirname(file_to), exist_ok=True)
        shutil.copyfile(f, file_to)


def set_network(cfg, num_classes):
    if 'B0' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2B0(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'B1' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2B1(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'B2' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2B2(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'B3' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2B3(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'S' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2S(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'M' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2M(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'L' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2L(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    elif 'XL' == cfg.network:
        model = keras_efficientnet_v2.EfficientNetV2XL(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=cfg.dropout,
            num_classes=num_classes,
            pretrained=cfg.pretrained)
    else:
        print('Use EfficientNetV2B0')
        model = keras_efficientnet_v2.EfficientNetV2B0(
            input_shape=(cfg.img_size, cfg.img_size, 3),
            dropout=1e-6,
            num_classes=num_classes,
            pretrained=cfg.pretrained)

    return model


def train(cfg):
    # Initial settings
    datagen_kwargs = dict(rescale=1. / 255)
    dataflow_kwargs = dict(target_size=(cfg.img_size, cfg.img_size),
                            batch_size=cfg.batch_size,
                            interpolation="bilinear")
    # Creation of validation data
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        cfg.val_data_dir, shuffle=False, **dataflow_kwargs)

    # Creation of training data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    train_generator = train_datagen.flow_from_directory(
        cfg.train_data_dir, shuffle=True, **dataflow_kwargs)

    # Model settings
    model = set_network(cfg, train_generator.num_classes)
    model.compile(loss=cfg.loss,
                  optimizer=cfg.optimizer,
                  metrics=['acc'])
    model.summary()

    # Call back settings
    os.makedirs(os.path.dirname(cfg.output_model), exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        cfg.output_model,
        monitor='val_loss', verbose=1, save_best_only=True,  # accuracy  val_loss
        save_weights_only=True, mode='auto', save_freq='epoch')  #save_freq='epoch'

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    # Training
    history = model.fit(train_generator,
                        epochs=cfg.epoch,
                        validation_data=valid_generator,
                        callbacks=[cp_callback, es_callback, tensorboard_callback],
                        )


def predict_alt(cfg):
    # Initial settings
    datagen_kwargs = dict(rescale=1. / 255)
    dataflow_kwargs = dict(target_size=(cfg.img_size, cfg.img_size),
                            batch_size=cfg.batch_size,
                            interpolation="bilinear")

    # Load test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    test_generator = test_datagen.flow_from_directory(
        cfg.test_alt_data_dir, shuffle=False, **dataflow_kwargs)

    # Correct labels
    y_true = test_generator.classes

    # Load trained model
    model = set_network(cfg, 3)
    model.load_weights(cfg.output_model)

    # Predict labels
    predicts = model.predict(test_generator)
    y_pred = np.argmax(predicts, axis=1)

    org_gen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs).flow_from_directory(
        cfg.test_data_dir, shuffle=False, **dataflow_kwargs)

    data_dict = {"filepath":test_generator._filepaths}
    data_dict.update({k:predicts[:,v] for k,v in org_gen.class_indices.items()})
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(os.path.dirname(cfg.output_model), "test_preds.csv"), index=False)


"""
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
'__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__',
'__module__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
'__str__', '__subclasshook__', '__weakref__', '_filepaths', '_flow_index', '_get_batches_of_transformed_samples',
'_keras_api_names', '_keras_api_names_v1', '_set_index_array', 'allowed_class_modes', 'batch_index', 'batch_size',
'class_indices', 'class_mode', 'classes', 'color_mode', 'data_format', 'directory', 'dtype', 'filenames', 'filepaths',
'image_data_generator', 'image_shape', 'index_array', 'index_generator', 'interpolation', 'keep_aspect_ratio', 'labels',
'lock', 'n', 'next', 'num_classes', 'on_epoch_end', 'reset', 'sample_weight', 'samples', 'save_format', 'save_prefix',
'save_to_dir', 'seed', 'set_processing_attrs', 'shuffle', 'split', 'subset', 'target_size', 'total_batches_seen', 'white_list_formats']
"""

def main():
    # Loading configuration files
    with open('config.yaml') as file:
        cfg = AttrDict(yaml.safe_load(file))
        print('param:')
        for k in cfg.keys():
            print(f'{k}: {cfg[k]}')

    if cfg.run_type == 'train':
        train(cfg)
    elif cfg.run_type == 'predict':
        predict(cfg)
    elif cfg.run_type == 'predict_alt':
        predict_alt(cfg)
    else:
        print(f'Differ run_type')


if __name__ == '__main__':
    main()
