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
