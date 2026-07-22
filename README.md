# vrc-autopilot

[![PyPI](https://img.shields.io/pypi/v/vrc-autopilot)](https://pypi.org/project/vrc-autopilot/)
[![Python](https://img.shields.io/badge/python-3.14-blue)](https://pypi.org/project/vrc-autopilot/)
[![License](https://img.shields.io/pypi/l/vrc-autopilot)](LICENSE)

[VRCPositionHUD](https://github.com/njm2360/vrc-position-hud)を利用してOSCで移動・視点を操作する自動化ツール

> [!WARNING]
> 本ツールの使用によって生じたいかなる結果についても、作者は一切の責任を負いません。自己責任で使用してください。
> 使用にあたっては各ワールド・コミュニティのルールに従ってください。自動化を歓迎しない場所では使わないこと。

- 全体像とモジュール対応: [docs/architecture.md](docs/architecture.md)
- プラント特性の測定手順: [docs/system-identification.md](docs/system-identification.md)
- 制御ゲインの根拠と調整手順: [docs/gain-tuning.md](docs/gain-tuning.md)
- オフライン検証の組み方: [docs/verification.md](docs/verification.md)
- 各コマンドの使用方法: [docs/usage.md](docs/usage.md)

## 動作環境

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## インストール

```sh
uv add vrc-autopilot
```

## サンプルコード

移動・照準・押下は `Pilot` API で記述する。マップ・プラント・ボタン座標を渡して巡回ルートを組む。

- [でかプ 軽量化スイッチ自動化](examples/dekapu/main.py)
    ※マップデータ同梱

## CLI

`uv run <コマンド>` で実行する。フラグ詳細は各 `--help` で確認可能

| コマンド          | 用途                                                                         |
| ----------------- | ---------------------------------------------------------------------------- |
| `decode-demo`     | HUD を読み取り 6DoF を表示(動作確認とキャリブレーション)                     |
| `map-room`        | 壁沿いに歩いて部屋マップを記録                                               |
| `find-button`     | 複数地点からボタンを三角測量                                                 |
| `probe-axes`      | 入力軸の応答特性を測って `plant.json` に同定(制御ゲインの前提)               |
| `calibrate-world` | ワールドごとに変わる移動速度を測り、ゲインの倍率を補正                       |
| `bode-margins`    | 同定プラント上で全制御ループの安定余裕(ωc/PM/GM)とボード線図を出す(実機不要) |
| `log-video`       | 制御ログCSVを一人称3D+2D地図の動画(mp4)に再生                                |
