# HUD ビットグリッドのプロトコル定数。hlsl/PoseTelemetryHUD.shader と一致必須。
MAGIC = 0x5AC3E7A1

# ワード配列(uint32[12])のレイアウト
IDX_MAGIC = 0
IDX_TIME = 1
IDX_POS = slice(2, 5)  # x, y, z
IDX_FWD = slice(5, 8)  # forward x, y, z
IDX_UP = slice(8, 11)  # up x, y, z
IDX_CHECKSUM = 11
WORD_COUNT = 12

# グリッド
OFFSET_X = 8  # _OffsetX: グリッド左上のXオフセット
OFFSET_Y = 8  # _OffsetY: グリッド左上のYオフセット
BLOCK = 4  # _BlockPx: 1ビットの一辺。白=1, 黒=0
ROWS = WORD_COUNT
COLS = 32

GRID_W = COLS * BLOCK  # 64
GRID_H = ROWS * BLOCK  # 48

# キャプチャ切り出し領域
CAPTURE_MARGIN = 8
CAPTURE_W = OFFSET_X + GRID_W + CAPTURE_MARGIN  # 80
CAPTURE_H = OFFSET_Y + GRID_H + CAPTURE_MARGIN  # 64

# RGB和の二値化しきい値
THRESHOLD = 3 * 128
