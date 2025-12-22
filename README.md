了解です。
**「今の実装仕様（environment_config.py が Python / 事前計算あり / main.py はランチャ）」**に合わせて、README を**研究用途としてそのまま配布できるレベル**まで整理して書き直します。

以下は **修正版 README.md（全文差し替え想定）** です。
そのまま `README.md` に貼り替えて使えます。

---

# Projector Warp System

**360°スクリーン対応・マルチプロジェクター歪み補正投影システム**

---

## 🎯 概要

本システムは、

* **360°スクリーン**
* **複数プロジェクター**
* **凸面鏡（反射投影）**
* **Blenderで作成した物理モデル**

を用いた環境において、
**歪み補正済みの映像をリアルタイムに表示するための Python ベース投影システム**です。

### 特徴

* Blender 上で作成した **スクリーン・鏡・プロジェクター配置**を使用
* 幾何情報をもとに **事前計算（Warp Map 生成）**
* 実行時は **軽量・高速描画**
* GUI による **補正方式切替・グリッド編集**
* 複数ディスプレイ・オーバーラップ領域対応（ブレンド）

---

## 🛠️ 開発環境

* エディタ：VS Code
* ターミナル：PowerShell（VS Code 統合）
* DCC ツール：Blender
* OS：Windows 10 / 11（複数ディスプレイ必須）

---

## 📦 システム全体の流れ（重要）

```
① Blender で投影環境をモデリング
        ↓
② environment_config.py を生成・保存
        ↓
③ 事前計算スクリプトで Warp Map を生成
        ↓
④ main.py を起動（編集・表示）
        ↓
⑤ media_player_multi.py が補正表示を実行
```

👉 **実行時に重い計算は行いません**
👉 時間がかかる処理はすべて「事前計算」に切り出されています

---

## 🧩 ディレクトリ構成

```
Sotsuken/
│  LICENSE.txt
│  README.md
│
│  main.py                  # GUIランチャ（編集・表示制御）
│  media_player_multi.py    # 補正済み映像のマルチ出力
│  warp_engine.py           # 歪み補正コア（OpenCV / CUDA）
│  precompute_warp_maps.py  # 事前計算（Warp Map生成）
│
├─config
│  │  edit_profile.json
│  │  environment_config.py   # Blender由来の物理環境定義（重要）
│  │
│  └─projector_profiles
│        __._DISPLAY2_perspective_points.json
│        __._DISPLAY3_warp_map_points.json
│
├─editor
│     grid_editor_perspective.py
│     grid_editor_warpmap.py
│     grid_utils.py
│
└─temp
       editor_active_*.lock
```

---

## 🧱 Blender → Python への事前準備（重要）

### 1️⃣ Blender でモデルを作成

Blender 上で以下を **正確なスケール・位置関係**で作成します。

* スクリーン（360°曲面）
* 凸面鏡
* プロジェクター（位置・向き・FOV）

### 推奨ルール

* 単位：**メートル**
* Z軸：上方向
* 原点：床中心 or 観察者基準

---

### 2️⃣ Blender から情報を Python に書き出す

Blender の Python から以下を取得し、
**Pythonファイルとして保存**します。

📄 `config/environment_config.py`

```python
# Auto-generated from Blender

environment_config = {

    "screen_simulation_sets": [
        {
            "name": "ScreenSimulatorSet_1",

            "projector": {
                "origin": [1.2, -3.4, 2.1],
                "direction": [0.0, 1.0, -0.1],
                "fov_h": 90.0,
                "fov_v": 60.0,
                "resolution": [1920, 1080]
            },

            "mirror": {
                "vertices": [
                    [0.5, 0.2, 1.1],
                    [0.6, 0.2, 1.1],
                    ...
                ]
            },

            "screen": {
                "material": "Screen_mat",
                "vertices": [
                    [2.1, 3.0, 1.5],
                    [2.2, 3.1, 1.6],
                    ...
                ]
            }
        },

        # 複数プロジェクター分
    ],

    "observer": [0.0, 0.0, 1.65]
}
```

### 🔴 注意

* **JSONではありません**
* Python の `dict` として **直接 import されます**
* 10万行規模でも問題ありません（事前計算専用）

---

## ⚙️ 事前計算（必須）

### Warp Map を事前生成

```powershell
python precompute_warp_maps.py
```

この処理で：

* Blenderモデル
* プロジェクター
* 鏡
* スクリーン

を使って **ピクセル単位の歪み補正マップ**を生成します。

💡 この処理は **重いが1回だけ**

---

## 🖥 実行（main.py）

事前計算が終わったら、通常の起動はこちら：

```powershell
python main.py
```

### main.py の役割

* 編集用ディスプレイの自動認識
* 補正方式の選択
* グリッド編集の起動
* 補正表示の起動（事前計算結果を使用）

👉 **main.py 自体は重い計算をしません**

---

## 🛠 補正モード

### perspective

* 4点射影変換
* 平面・簡易構成向け
* 軽量・高速

### warp_map

* 多点グリッド（例：6×6）
* 曲面スクリーン対応
* オーバーラップ・ブレンド可能

---

## ⏱ 実行時の速度について

| 処理               | 実行タイミング | 時間     |
| ---------------- | ------- | ------ |
| Blender → Python | 事前      | 人力     |
| Warp Map 計算      | 事前      | 数秒〜数分  |
| main.py 起動       | 実行時     | 即時     |
| 補正表示             | 実行時     | リアルタイム |

✔ **事前計算が終わっていれば、起動は速い**

---

## 🖥 ハードウェア構成（例）

* CPU：Intel Core i9
* GPU：RTX 4070 Ti SUPER
* メモリ：32GB
* プロジェクター：Optoma GT1080 ×3
* 編集用モニター：1台

---

## ⚠️ 注意点まとめ

* `environment_config.py` は **JSONではない**
* 事前計算を忘れると起動時に遅くなる
* 新しい Blender モデルを使ったら **必ず再計算**
* 複数ディスプレイ必須

---

## 📄 ライセンス

Custom License（研究用途）

---

## 💡 設計思想（補足）

* **重い処理はすべてオフライン**
* 実行時は **GPU転送＋描画のみ**
* Blender を「設計ツール」として使う
* Python は「実行エンジン」

---
