from dataclasses import dataclass

MAGIC = 0x5AC3E7A1

# ワード(行)のインデックス定義
IDX_MAGIC = 0
IDX_TIME = 1
IDX_POS = slice(2, 5)  # x, y, z
IDX_FWD = slice(5, 8)  # forward x, y, z
IDX_UP = slice(8, 11)  # up x, y, z
IDX_CHECKSUM = 11
WORD_COUNT = 12


@dataclass(frozen=True)
class GridSpec:
    """ビットグリッドの幾何とプロトコル定数。

    シェーダー側 _OffsetX/_OffsetY/_BlockPx と一致させること。
    """

    offset_x: int = 8  # _OffsetX: グリッド左上のXオフセット(px)。注入可能
    offset_y: int = 8  # _OffsetY: グリッド左上のYオフセット(px)。注入可能
    block: int = 4  # _BlockPx: 1ビットの一辺(px)。白=1, 黒=0。注入可能
    rows: int = 12  # 行数 = ワード数
    cols: int = 32  # 列数 = 1ワードのビット幅(MSBが左端)
    magic: int = MAGIC

    # キャプチャ切り出し領域(クライアント左上基準)。None ならオフセット/ブロックから
    # グリッドを内包するサイズを自動導出する(offset/block を注入しても破綻しない)。
    capture_w: int | None = None
    capture_h: int | None = None
    capture_margin: int = 8  # 自動導出時のグリッド外周マージン(px)

    # 白黒判定しきい値(RGB合計)。純白(765)/純黒(0)しか出ない前提の中間値。
    threshold: int = 3 * 128

    def __post_init__(self) -> None:
        if self.rows != WORD_COUNT:
            raise ValueError(
                f"rows must be {WORD_COUNT} (1 word per row), got {self.rows}"
            )
        if self.cols != 32:
            raise ValueError(f"cols must be 32 (32-bit words), got {self.cols}")
        if self.offset_x < 0 or self.offset_y < 0:
            raise ValueError("offsets must be non-negative")
        if self.block < 1:
            raise ValueError("block must be >= 1")
        # capture 未指定なら幾何から自動導出(frozen なので object.__setattr__)
        if self.capture_w is None:
            object.__setattr__(
                self, "capture_w", self.offset_x + self.grid_w + self.capture_margin
            )
        if self.capture_h is None:
            object.__setattr__(
                self, "capture_h", self.offset_y + self.grid_h + self.capture_margin
            )
        if (
            self.offset_x + self.grid_w > self.capture_w
            or self.offset_y + self.grid_h > self.capture_h
        ):
            raise ValueError("capture region does not contain the grid at its offset")

    @property
    def grid_w(self) -> int:
        """グリッド全体の幅(px)。仕様では 64。"""
        return self.cols * self.block

    @property
    def grid_h(self) -> int:
        """グリッド全体の高さ(px)。仕様では 24。"""
        return self.rows * self.block


DEFAULT_SPEC = GridSpec()
