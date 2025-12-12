## 🛠️ **開発環境**
* エディタ：VS Code
* ターミナル：PowerShell（VS Code統合ターミナル）
* Blender（この研究で使うスクリーンのモデル）

## 📝 `README.md`（プロジェクターゆがみ補正＆マルチスクリーン統合投影システム）

# Projector Warp System
360°スクリーン＋複数プロジェクター＋凸面鏡構成において、歪み補正済みのPC画面を自然に表示するためのPythonベースのプロジェクションシステムです。

## 🎯 概要
このアプリケーションは、以下の構成を持つプロジェクション環境に対応しています：

- 複数のプロジェクター（任意の台数）
- 凸面鏡経由の360°スクリーン投影
- 各ディスプレイごとの歪み補正（射影変換 or 自由変形）
- モード切替に応じた自動初期グリッド生成と編集
- 重なり部分のアルファブレンディング対応（左右10°）
- JSONファイルから情報を得てレイトレーシング

## 📦 動作の流れ
- 自動的に編集用ディスプレイを認識
- GUIから補正モード（`perspective` / `warp_map`）を選択
- 各スクリーンに応じた最適な初期グリッドを自動生成
- グリッドのGUI編集（ズーム・パン・ドラッグ対応）
- 編集後、補正された画面をリアルタイム表示
- ウィンドウや動画、ゲームも自然な表示に対応

## 🧩 ディレクトリ構成

```
Sotsuken/
│  LICENSE.txt　#ライセンス
│  main.py　# 起動＆UI・編集／表示操作
│  media_player_multi.py　# 各プロジェクターへ補正表示
│  percentage.py　# ピクセル使用率の確認
│  README.md　# 説明書
│  warp_engine.py　# 歪み補正ロジック（OpenCV）
│  
├─config
│  │  edit_profile.json　# 編集用ディスプレイ情報
│  │  environment_config.py　# スクリーン・鏡・プロジェクター定義
│  │
│  └─projector_profiles　# 各ディスプレイ用グリッド保存
│         __._DISPLAY2_perspective_points.json
│         __._DISPLAY3_perspective_points.json
│         __._DISPLAY4_perspective_points.json
│
├─editor
│     grid_editor_perspective.py　# perspectiveグリッド編集GUI
│     grid_editor_warpmap.py　# warp_mapグリッド編集GUI
│     grid_utils.py　# グリッド点生成の共通関数
│
└─temp　# エディター自動終了用ロックファイル

```

## 🖥 使用方法

### 1. 起動
```bash
python main.py
```

### 2. 操作手順

1. 起動すると「編集用ディスプレイ（メイン）」が自動認識されます。
2. 「補正方式を選択」から `perspective` か `warp_map` をプロジェクションマッピング対象の物体に合わせて選びます。
3. 「グリッドエディター起動」を押すと、チェックボックスで指定したスクリーンのグリッド編集ウィンドウが起動。
4. 点をドラッグして微調整し、閉じれば自動保存。

#### ⚠️ 注意：
新しくグリッドエディターを開くと、既存のグリッドは初期状態に戻ります。編集済みのグリッドがある場合は必ず保存・閉じた後に新しいエディターを起動してください。

5. 「補正表示起動」を押すと、補正済みの画面がプロジェクター側に表示されます。

## 🛠 補正モード

### - perspective

* 4点の射影変換を利用
* 画面の歪みを補正して長方形に投影

### - warp\_map

* 外周グリッドを自由変形しマッピング
* 曲面スクリーン・オーバーラップ領域に有効

* 自動保存・自動終了機能付き（ロックファイルで制御）

## ⚙️ 環境設定ファイル：`settings/config/environment_config.py`

```python
environment_config = {
  "screen_simulation_sets": [
    {
      "name": "ScreenSimulatorSet_1",
      "projector": {
        "origin": [...],
        "direction": [...],
        "fov_h": ...,
        "fov_v": ...,
        "resolution": [...]
      },
      "mirror": {
        "vertices": [ [...], [...] ]
      },
      "screen": {
        "material": "Screen_mat",
        "vertices": [ [...], [...] ]
      }
    },

    ''' 3セット分 '''

  ],

  "observer": [x, y, z]
}

```

## 🎮 対応する表示コンテンツ

* Unityによる360度コンテンツ
* YouTubeやVLCでの動画再生
* 任意のPCウィンドウ（ブラウザ・ゲームなど）

## 🖥️ **ハードウェア構成**

* **PCスペック**：
  * CPU: Intel Core i9
  * GPU: GeForce RTX 4070 Ti SUPER
  * メモリ: DDR5 32GB (16GB x2)
  * ストレージ: M.2 SSD 2TB

* **表示機器**：
  * プロジェクター：オプトマ GT1080 x 3台（表示用）
  * モニター：1台（編集用）

## 📌 必要環境

* Python 3.x.x
* OpenCV
* PyQt5
* tkinter
* numpy
* Windows 10/11 64bit（複数ディスプレイ必須）

## ⚠️ 注意点

* 起動時に各ディスプレイにポイントファイルがない場合、自動生成されます。
* 編集後は必ずウィンドウを閉じるか、ロックファイルを削除して自動保存・終了させてください。
* 新しいグリッドエディターを開くと既存のグリッドは初期状態に戻ります。編集済みグリッドは必ず保存してください。
* すべての表示はリアルタイムにリマップされるため、マシンスペックに依存します。

## 📄 ライセンス

custom License

---

# 追加機能について

## 🛠️ **GPU対応とCUDAビルドに関する設定**

### 💻 GPU使用条件
本プロジェクトでは、OpenCV の CUDA モジュールおよび CuPy を使用することで、GPU 上で高速に歪み補正処理を実行できます。ただし、以下の条件が揃っている必要があります：

- NVIDIA GPU が搭載されていること  
  （RTX 4070 Ti SUPER など、CUDA 対応 GPU）  
- GPU ドライバーが対応する CUDA バージョンをサポートしていること  
  （例：CUDA 12.3 で CuPy 13.2、OpenCV 4.x の CUDA ビルド）  
- OpenCV が CUDA サポート付きでビルドされていること  

#### ⚠️ 注意：
- ドライバーや CUDA バージョンが不適合の場合、CPU モードで動作します。  
- OpenCV の Python パッケージ（`opencv-python`）は通常 CPU ビルドのため、GPU 処理には **CUDA ビルド** が必須です。

---

### 🏗 CUDA ビルドが必要になる条件
以下の条件に当てはまる場合は、OpenCV を CUDA 対応でビルドしてください：

1. `cv2.cuda.getCudaEnabledDeviceCount()` が 0 を返す  
2. 歪み補正処理をリアルタイムで行いたい  
3. 複数ディスプレイや高解像度（1920x1080 × 複数）での出力が遅い  

---

### ⚙️ CUDA ビルド手順（Windows + Python）

1. **依存ライブラリのインストール**
```powershell
# 事前に管理者権限のpowershellで以下を実行
Set-ExecutionPolicy Bypass -Scope Process -Force`
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 入った来夏で確認し、バージョンが表示されれば次を実行
choco --version

# 必要なビルドツール
choco install cmake
choco install git
```

####　※この時点での注意事項
cmakeがインストールされていなかった場合、以下のリンクからインストールしてください。
https://cmake.org/download/

2. **OpenCV ソースコードの取得**

```powershell
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

3. **CMake で CUDA ビルド設定**

```powershell
cd opencv
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ^
      -A x64 ^
      -D CMAKE_BUILD_TYPE=Release ^
      -D CMAKE_INSTALL_PREFIX=..\install ^
      -D OPENCV_EXTRA_MODULES_PATH=..\opencv_contrib\modules ^
      -D WITH_CUDA=ON ^
      -D CUDA_ARCH_BIN=86 ^             # RTX 4070 Ti に対応
      -D BUILD_opencv_python3=ON ^
      -D BUILD_EXAMPLES=OFF ..
```

4. **ビルド & インストール**

```powershell
cmake --build . --config Release --target INSTALL
```

5. **Python 環境にインストール**

```powershell
# install フォルダ内の cv2.pyd を仮想環境にコピーするか、
# PYTHONPATH に追加
set PYTHONPATH=%CD%\install\python;%PYTHONPATH%
```

---

### ✅ 動作確認

```python
import cv2
import cupy

# CuPy デバイス確認
print("CuPy devices:", cupy.cuda.runtime.getDeviceCount())

# OpenCV CUDA デバイス確認
print("cv2.cuda devices:", cv2.cuda.getCudaEnabledDeviceCount())
```

* 両方とも 1 以上を返せば GPU 処理が可能です
* `0` を返す場合は、CUDA ビルドやドライバー、CUDA バージョンを再確認してください

---

### ⚡ GPU 利用上の注意

* GPU 使用時は、他のアプリケーション（ブラウザ、VSCode など）が GPU メモリを占有していると、CUDA メモリ不足で CPU fallback になる場合があります
* 複数ディスプレイ投影時は GPU メモリ消費が大きくなるため、RTX 4070 Ti 以上を推奨
* CuPy + OpenCV CUDA は独立して GPU メモリを使用するため、併用時は余裕を持たせること

---

💡 **まとめ**

* CPU モードでの動作も可能だが、リアルタイム性は落ちる
* GPU を有効にするには OpenCV の CUDA ビルドと対応 GPU ドライバーが必須
* CuPy は簡単に GPU 処理が確認できるため、まず CuPy が GPU を認識するかチェックすると良い
