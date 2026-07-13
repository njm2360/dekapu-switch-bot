# アーキテクチャ概要

VRChat 自動化のデータフローと、それを実装する `pose_hud` パッケージ / CLI の対応。

## パイプライン

```
VRChat 画面(HUDビットグリッド)
    │  スクリーンキャプチャ(ネイティブ解像度・クライアント左上の小領域)
    ▼
[capture]  WindowsVRChatCapture ──► フレーム(numpy)
    │
    ▼
[decode]   decode_pose ──► Pose(位置・前方・上方・時刻)+ 検証
    │
    ▼
[reader]   PoseReader(スレッドで読み続ける・統計・コールバック/ジェネレータ)
    │
    ├─► [mapping]     RoomMapper ──► 部屋の地図(占有グリッド/間取り図)
    ├─► [triangulate] Sighting/triangulate ──► ボタンのワールド座標(最小二乗交点)
    └─► [navigation]  NavGrid + plan_path ──► 壁を避けた経路
                          │
                          ▼
                     [control] PID ──► [osc] VRChatOSC ──► /input/* をVRChatへ注入
```

## モジュール(`pose_hud/`)

| モジュール          | 役割                                                            |
| ------------------- | --------------------------------------------------------------- |
| `spec.py`           | `GridSpec`。グリッド/プロトコル定数の一元管理(シェーダーと一致) |
| `decode.py`         | numpy ベクトル化デコード + 検証(`decode_pose`, `Pose`)          |
| `encode.py`         | 合成エンコーダ(`render_pose`)。テスト/キャリブレーション用      |
| `capture.py`        | Windows/VRChat ウィンドウキャプチャ(DPI対応、`FrameSource`)     |
| `reader.py`         | `PoseReader`(実運用API)。統計・デバッグダンプ                   |
| `mapping.py`        | `RoomMapper`。軌跡→寸法・占有グリッド・保存/読込(ペンアップ分割対応) |
| `mapping_render.py` | 間取り図の描画(matplotlib 任意依存)                             |
| `triangulate.py`    | 視線レイの最小二乗交点でボタン座標を推定                        |
| `navigation.py`     | 歩行可能グリッド生成(壁回避)+ A* 経路計画 + 照準誤差            |
| `control.py`        | 汎用 PID(視点・速度のフィードバック制御)                        |
| `osc.py`            | VRChat への OSC 送信(移動・視点・アバターパラメータ)            |

## CLI(`pose_hud/cli/`, console scripts)

| コマンド         | スクリプト              | 用途                                                       |
| ---------------- | ----------------------- | ---------------------------------------------------------- |
| `decode-demo`    | `cli/decode_demo.py`    | HUD を読み取り 6DoF を表示(動作確認)                       |
| `map-room`       | `cli/map_room.py`       | 壁沿いに歩いて部屋マップを記録(SPACE一時停止・日時フォルダ出力) |
| `find-button`    | `cli/find_button.py`    | 複数地点からボタンを三角測量(SPACE/r/q)                    |
| `patrol-buttons` | `cli/patrol_buttons.py` | マップ上でボタンを壁を避けて巡回(OSC + PID)                |

## 確定仕様

グリッド/プロトコルの確定仕様は [pose-telemetry-hud-spec.md](pose-telemetry-hud-spec.md) を参照。
デコーダはこれに完全準拠し、定数の変更はオーナー確認を要する。
