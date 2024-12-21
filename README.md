# jupyter-env

## 環境
MacBook Air M2 arm64

## minicondaのインストール
```bash
brew install miniconda
conda -V # コマンドが通ることを確認
conda init zsh 
source ~/.zshrc
```

## Conda上に仮想環境を作成
```bash
conda create -n {環境名} python=3.9
e. g. 
conda create -n jupyter # pythonのバージョンはこだわりがなければ省略OK
※最新のpythonのバージョンだとライブラリがうまく動かないかも
```

## Condaの設定
```bash
conda activate jupyter # 作成した仮想環境に入る
conda install -c conda-forge jupyterlab # Jupyter Labをインストール
```

## 必要なライブラリをインストール
googleで「conda pytorch」など「conda {ライブラリ名}」で検索するとAnacondaのWebページが出るのでそれを参考にする。
https://anaconda.org/pytorch/pytorch
```bash
conda install pytorch::pytorch
```

## Jupyter Labの起動
下記のコマンドで、ブラウザ上にjupyterが立ち上がります。
```bash
jupyter lab
```

## GPUの使用確認 ( Pytorch )
下記の通りに実行する。
```python
import torch
device = torch.device('mps')
torch.backends.mps.is_available()
# Trueが返ればOK
```

## GPUの使用確認 ( Keras )
下記の通りに実行する。
ライブラリのインストール
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

Jupyter Labの起動
```bash
jupyter lab
```

GPUの使用の確認
```python
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Is GPU available?", tf.config.list_physical_devices('GPU'))
```
下記が返る
```bash
TensorFlow version: 2.16.2
Is GPU available? [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
適当にコードを実行する
```python3
import tensorflow as tf
import numpy as np

# 簡単なデータセットとモデルを作成
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# トレーニング
model.fit(x_train, y_train, epochs=5)

# 評価
model.evaluate(x_test, y_test)
```


