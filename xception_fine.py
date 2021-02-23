import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input


rate = 0.0001
epochs = 50
task = 'top'    # 'fine'
model_name = 'compare'
base_model = 'xception'
initial_epoch = 0
run_test = True


model_path = f"{model_name}.h5"
tensorboard_dir = f"tb"
data_path = f"shots.tsv"

views_min = 1000

test_part = 0.2
train_part = 0.75
shape = (299, 299, 3)
batch_size = 100
steps_epoch = 1000
steps_val = 250
steps_test = 250


def log(message):
    print(dt.now().strftime('%Y-%m-%d %H:%M:%S') + f': {message}\n', end='', flush=True)


log('model...')

if task.startswith('top'):
    if base_model.lower() == 'xception':
        base = Xception(include_top=False,
                        weights='imagenet',
                        input_shape=shape,
                        pooling='avg')
    else:
        base = next(lr for lr in keras.models.load_model(f"{base_model}.h5").layers
                    if 'xception' in lr.name.lower())

    inputs = [Input(shape=shape), Input(shape=shape)]

    compare = keras.layers.subtract([base(i) for i in inputs])
    compare = Dense(256, activation='relu')(Dropout(0.25)(compare))
    compare = Dense(64, activation='relu')(Dropout(0.5)(compare))
    compare = Dense(1, activation='sigmoid', name='compare')(Dropout(0.5)(compare))

    model = Model(inputs, compare)

    base.trainable = False

else:
    model = keras.models.load_model(model_path)

    base = next(lr for lr in model.layers if 'xception' in lr.name.lower())

    base.trainable = True
    trainable = False
    for lr in base.layers:
        if not trainable and 'block13' in lr.name.lower():
            trainable = True

        lr.trainable = trainable

model.summary()


if initial_epoch != 0:
    model.load_weights(f"{model_name}.checkpoint")

    if initial_epoch < 0:
        model.save(model_path, save_format='h5')
        log('saved')
        exit(0)


metrics = [keras.metrics.AUC()]
metrics.extend(keras.metrics.Precision(thresholds=t, name=f"pre{t}") for t in [0.5, 0.6, 0.75, 0.9])
metrics.extend(keras.metrics.Recall(thresholds=t, name=f"re{t}") for t in [0.5, 0.6, 0.75, 0.9])

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.RMSprop(learning_rate=rate),
              metrics=metrics)


log('data...')

data = pd.read_csv(data_path, sep='\t')


video_ctr = (data
             .groupby('video_id')
             .agg({k: 'sum' for k in ['total_views', 'total_clicks']})
             .reset_index(drop=False))

data.drop(['total_views', 'total_clicks'], axis=1, inplace=True)
data = data.merge(video_ctr, how='inner', on='video_id')

data = data.query(f"total_views >= {views_min}").reset_index(drop=True)
data['total_ctr'] = data['total_clicks'].fillna(0.0).abs() / data['total_views']


video_stat = data.groupby('video_id').agg({'filepath': 'size'})
video_stat.columns = ['size']
video_stat = video_stat[video_stat['size'] > 1].reset_index(drop=False)
data = data.merge(video_stat, how='inner', on='video_id').reset_index(drop=True)


data['value'] = data['total_ctr']


random_state = np.random.RandomState(seed=42)
video_test = (video_stat['video_id']
              .sample(frac=test_part, random_state=random_state).values)
video_train = (video_stat[~video_stat['video_id'].isin(video_test)]['video_id']
               .sample(frac=train_part, random_state=random_state).values)

data.loc[data['video_id'].isin(video_test), 'part'] = 'test'
data.loc[data['video_id'].isin(video_train), 'part'] = 'train'
data.loc[data[data['part'].isna()].index, 'part'] = 'val'

part_sizes = data['part'].value_counts()

log(f"\n{part_sizes}")


def load(path, attempts=10):
    for _ in range(attempts):
        try:
            im = img_to_array(load_img(path, target_size=shape[:2]))
            if im.shape != shape:
                raise ValueError(f"wrong shape {im.shape}")

            return preprocess_input(im)

        except Exception as e:
            log(f"image error: {e}")

    raise RuntimeError('wrong data')


def gen(df):
    log(f"gen {len(df)}...")

    if len(df) == 0:
        raise ValueError('data is empty')

    video_ids = df['video_id'].unique()
    ids = {}
    for video_id in tqdm(video_ids):
        ids[video_id] = df[df['video_id'] == video_id].index.values

    def pair_(same=True, strict=True):
        if same:
            return tuple(np.random.choice(ids[np.random.choice(video_ids)], size=2, replace=True))

        return tuple(np.random.choice(ids[i]) for i in np.random.choice(video_ids, size=2, replace=not strict))

    def gen_():
        while True:
            left, right, target = [], [], []
            for f, s in (pair_(same=False) for _ in range(batch_size)):
                left.append(load(df.loc[f, 'filepath']))
                right.append(load(df.loc[s, 'filepath']))
                target.append(float(df.loc[f, 'value'] > df.loc[s, 'value']))

            yield [np.array(left), np.array(right)], np.array(target)

    log(f"gen {len(df)}: done")

    return gen_()


log('fit...')


model.fit(x=gen(data.query("part == 'train'")),
          epochs=epochs,
          steps_per_epoch=steps_epoch,
          validation_data=gen(data.query("part == 'val'")),
          validation_steps=steps_val,
          initial_epoch=initial_epoch,
          verbose=1,
          callbacks=[keras.callbacks.TensorBoard(log_dir=f"{tensorboard_dir}/{task}"),
                     keras.callbacks.ModelCheckpoint(f"{model_name}.checkpoint",
                                                     monitor='val_loss',
                                                     save_best_only=True,
                                                     save_weights_only=True)])

model.save(model_path, save_format='h5')


if run_test:
    log('test...')

    score = model.evaluate(gen(data.query("part == 'test'")),
                           steps=steps_test,
                           verbose=1,
                           return_dict=True)

    log(f"score: {score}")


log('done')
