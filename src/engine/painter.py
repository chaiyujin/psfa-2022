from functools import lru_cache
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assets import PATH_DEFAULT_FONT
from src.data import image as imutils


@lru_cache(maxsize=8)
def get_default_font(font_size: int = 20) -> ImageFont.ImageFont:
    return ImageFont.truetype(PATH_DEFAULT_FONT, font_size)


@lru_cache(maxsize=8)
def get_font(font_path, font_size: int = 20) -> ImageFont.ImageFont:
    return ImageFont.truetype(font_path, font_size)


class Text(object):
    def __init__(
        self,
        txt: str,
        pos: Union[Tuple[int, int], Tuple[float, float]],
        align: Tuple[str, str] = ("left", "top"),
        color: Union[Tuple[int, int, int], Tuple[float, float, float]] = (255, 255, 255),
        font: Optional[ImageFont.ImageFont] = None,
        font_size: Optional[int] = None,
    ):
        """
        Args:
            txt (str) : the text string.
            pos ((int, int) or (float, float)) : the position of anchor coordinate in canvas.
            align ((str, str)): the align method for anchor, first element is 'align_x' and second is 'align_y'.
                - 'align_x' can be 'left', 'center' or 'right';
                - 'align_y' can be 'top', 'middle' or 'bottom'.
            color ((int, int, int) or (float, float, float)): color in RGB format,
                if type is int, should be in range [0, 255];
                else type is float, should be in range [0, 1].
        """
        if font is None and font_size is not None:
            font = get_default_font(font_size)
        self.font = font
        # get pos and align
        (x, y), (ax, ay) = pos, align
        assert ax in ["left", "center", "right"], "unknown align x: {}".format(ax)
        assert ay in ["top", "middle", "bottom"], "unknown align y: {}".format(ay)
        self.txt = txt
        self.x, self.y = int(x), int(y)
        self.align_x, self.align_y = ax, ay
        # color
        clr_scale = 1
        if isinstance(color[0], float):
            assert all(0 <= x <= 1 for x in color), "float color not in range [0, 1]: {}".format(color)
            clr_scale = 255
        self.color = tuple(int(x * clr_scale) for x in color)
        assert all(0 <= x <= 255 for x in self.color), "color not in range [0, 255]: {}".format(self.color)


def put_texts(
    canvas: np.ndarray,
    text_tuples: Union[List[Text], Tuple[Text, ...], Text],
    font: Optional[ImageFont.ImageFont] = None,
    font_size: int = 20,
) -> np.ndarray:

    # auto convert to np.uint8
    is_float = False
    dtype = canvas.dtype
    if canvas.dtype in [np.float16, np.float32, np.float64]:
        is_float = True
        canvas = canvas * 255.0
    canvas = canvas.astype(np.uint8)

    # fmt: off
    assert canvas.dtype == np.uint8,\
        "put_texts(...) is only supported for np.uint8 canvas," " but {} is given".format(canvas.dtype)
    assert canvas.ndim == 3 and canvas.shape[-1] in [3, 4,],\
        "put_texts(...) only support (H, W, 3|4) canvas, " " but input canvas has shape with {}".format(canvas.shape)
    # fmt: on

    if font is None:
        font = get_default_font(font_size=font_size)

    # img_pil = Image.fromarray(canvas[..., :3])
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    ascent, descent = font.getmetrics()
    height = ascent + descent
    if isinstance(text_tuples, Text):
        text_tuples = (text_tuples,)
    for text in text_tuples:
        if text.txt is None:
            continue

        txt_font = text.font
        if txt_font is None:
            txt_font = font

        (width, _), _ = txt_font.font.getsize(text.txt)

        # get x, y
        x, y, ax, ay = text.x, text.y, text.align_x, text.align_y
        if ax == "left":
            pass
        elif ax == "center":
            x = x - width // 2
        elif ax == "right":
            x = x - width
        else:
            raise NotImplementedError()

        if ay == "top":
            pass
        elif ay == "middle":
            y = y - height // 2
        elif ay == "bottom":
            y = y - height
        else:
            raise NotImplementedError()

        draw.text((x, y), text.txt, font=txt_font, fill=text.color)
    # convert back to ndarray
    canvas = np.array(img_pil).astype(dtype)
    if is_float:
        canvas /= 255.0
    return canvas


def color_mapping(
    arr: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = "viridis", dtype=np.float32
) -> np.ndarray:

    # !important: don't import at top of file,
    #             because we have to set matplot.use in main.py
    import matplotlib.pyplot as plt

    arr = np.asarray(arr)
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
    cm = plt.get_cmap(cmap)
    # it's float
    img_data: np.ndarray = cm(np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1))[..., :3]
    # -> uint8
    if dtype in [np.uint8, "uint8"]:
        img_data = (img_data * 255.0).astype(dtype)
    return img_data


def plot_coeffs(name, coeffs, aspect=2.0, vmin=0, vmax=1):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(int(aspect * 5), 5))
    for i, x in enumerate(coeffs):
        plt.bar(i, x)
    plt.axis("off")
    plt.ylim(vmin, vmax)
    plt.title(f"{name} {vmin:.1f} ~ {vmax:.1f}", fontsize=20)
    plt.tight_layout()
    img = figure_to_numpy(fig)
    plt.close(fig)
    return img


def figure_to_numpy(fig) -> np.ndarray:
    # save it to a numpy array.
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def put_colorbar(
    colormap: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt="{:.1f}",
    unit: str = "",
    font_size=None,
    at: str = "right",
    txt_color=(0, 0, 0),
):
    assert at in ["right", "left", "top", "bottom"]
    dtype = colormap.dtype
    cm_pixel = np.copy(colormap)
    H, W = cm_pixel.shape[:2]
    A = min(H, W)

    if font_size is None:
        font_size = int(12.0 * A / 256.0)

    if at in ["right", "left"]:
        # colobar size
        cb_w = int(font_size * 0.9 * A / 256.0)
        cb_h = int(H / 2.5)
        # colorbar pixel
        cb_pixel = cb_h - 1 - np.arange(cb_h, dtype=np.float32)[:, None]  # [0, cb_h)
        cb_pixel = color_mapping(cb_pixel, cmap=cmap, vmin=0, vmax=cb_h - 1, dtype=dtype)
        cb_pixel = np.repeat(cb_pixel, cb_w, axis=1)
        cb_pixel = np.flip(cb_pixel, axis=0)
        cb_pixel = np.pad(
            cb_pixel, [(0, 0), (0, 0), (0, cm_pixel.shape[-1] - cb_pixel.shape[-1])], "constant", constant_values=255
        )

        # copy into place
        # - vertically middle
        y0 = H // 2 - cb_h // 2
        y1 = y0 + cb_h
        # - horizontally
        x0 = (W - cb_w - 1) if at == "right" else 1
        x1 = x0 + cb_w
        cm_pixel[y0:y1, x0:x1] = cb_pixel[..., [2, 1, 0]]

        # put text above and below the colorbar
        if at == "right":
            x = W
        else:
            x = 0
        # print(cb_w)
        # txt_color = (255, 255, 255)
        txts = [
            Text(fmt.format(vmin), (x - 25, y0), (at, "bottom"), txt_color),
            Text(fmt.format(vmax), (x - 25, y1), (at, "top"), txt_color),
            Text(unit, (x - 10, y1 + font_size * 0.8), (at, "top"), txt_color),
        ]
    else:
        # colobar size
        # cb_h = int(font_size * A / 256.0)
        cb_h = int(font_size * 1.2)
        cb_w = int(W / 4.0)
        # colorbar pixel
        cb_pixel = np.arange(cb_w, dtype=np.float32)[None, :]  # [0, cb_h)
        cb_pixel = color_mapping(cb_pixel, cmap=cmap, vmin=0, vmax=cb_w - 1, dtype=dtype)
        cb_pixel = np.repeat(cb_pixel, cb_h, axis=0)
        cb_pixel = np.pad(
            cb_pixel, [(0, 0), (0, 0), (0, cm_pixel.shape[-1] - cb_pixel.shape[-1])], "constant", constant_values=255
        )

        # copy into place
        # - horizontally middle
        x0 = W // 2 - cb_w // 2
        x1 = x0 + cb_w
        # - vertically
        y0 = (H - cb_h) if at == "bottom" else 0
        y1 = y0 + cb_h
        cm_pixel[y0:y1, x0:x1] = cb_pixel[..., [2, 1, 0]]

        # put text above and below the colorbar
        y = H if at == "bottom" else 0
        spacing = max(1, int(round(font_size * 0.1)))
        txts = [
            Text(fmt.format(vmin), (x0 - spacing, y), ("right", at), txt_color),
            Text(fmt.format(vmax) + unit, (x1 + spacing, y), ("left", at), txt_color),
        ]

    cm_pixel = put_texts(cm_pixel, txts, font_size=font_size)
    return cm_pixel


def heatmap(
    values: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar: bool = False,
    colorbar_at: str = "right",
    colorbar_fmt: str = "{:.1f}",
    colorbar_unit: str = "",
    dtype=np.float32,
):
    assert values.ndim == 2, "[heatmap()]: given values should be two-dim, but found shape {}".format(values.shape)

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    cm_pixel = color_mapping(values, cmap=cmap, vmin=vmin, vmax=vmax, dtype=dtype)[..., :3]

    if colorbar:
        cm_pixel = put_colorbar(
            cm_pixel, cmap=cmap, vmin=vmin, vmax=vmax, at=colorbar_at, unit=colorbar_unit, fmt=colorbar_fmt
        )

    return cm_pixel


def draw_canvas(
    imgs: List[List[Optional[np.ndarray]]],
    txts: List[List[Optional[Tuple[Any, ...]]]],
    A: int,
    titles: Optional[List[Optional[str]]] = None,
    shrink_columns: Union[Iterable[int], str] = (),
    shrink_ratio: float = 0.7,
):
    # check inputs
    assert len(imgs) == len(txts), "Given imgs and txts have different number of rows!"
    max_columns = 0
    for i, (row, txt) in enumerate(zip(imgs, txts)):
        assert len(row) == len(txt), "Given imgs and txts have different number of columns at row {}".format(i)
        max_columns = max(max_columns, len(row))
        for t in txt:
            assert t is None or len(t) % 2 == 0, "Given txt must be None or tuple of ('txt', color, ...)"

    # * process images
    def _shrink(img):
        w = int(A * min(shrink_ratio, 1))
        x0 = (A - w) // 2
        x1 = x0 + w
        return img[:, x0:x1]

    rows = []
    blank = np.zeros((A, A, 3), dtype=np.float32)
    for r in range(len(imgs)):
        padding = max_columns - len(imgs[r])
        # pad imgs
        for _ in range(padding):
            imgs[r].append(blank)
        for c in range(max_columns):
            if imgs[r][c] is None:
                imgs[r][c] = blank
            # make sure size and channels
            img = imutils.resize(imgs[r][c], (A, A))
            if img.ndim == 2:
                img = img[..., None]
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            img = img[..., :3]
            assert img.ndim == 3 and img.shape[-1] == 3
            # shrink
            if shrink_columns == "all" or c in shrink_columns:
                img = _shrink(img)
            # write back
            imgs[r][c] = img
        # concate columns
        rows.append(np.concatenate(imgs[r], axis=1))
    canvas = (np.concatenate(rows, axis=0) * 255.0).astype(np.uint8)

    # * fonts
    font_size = int(32 * A / 512.0)
    text_height = int(40 * A / 512.0)
    small_font_size = int(32 * A * 0.8 / 512.0)
    small_text_height = int(40 * A * 0.8 / 512.0)  # noqa: F841, not used

    # * process txts
    to_write = []
    y_acc = 0
    for r in range(len(rows)):
        x_acc = 0
        for img, txt in zip(imgs[r], txts[r]):
            if txt is not None:
                x = x_acc + img.shape[1] // 2
                y = y_acc
                for i_t in range(0, len(txt), 2):
                    to_write.append(
                        Text(
                            txt=txt[i_t],
                            pos=(x, y),
                            align=("center", "top"),
                            color=txt[i_t + 1] or (255, 255, 255),
                            font_size=font_size if i_t == 0 else small_font_size,
                        )
                    )
                    # accumulate text height for next line
                    y += text_height if i_t == 0 else small_text_height
            # accumuate col width
            x_acc += img.shape[1]
        # accumulate row height
        y_acc += rows[r].shape[0]
    canvas = put_texts(canvas, to_write, font_size=font_size)

    # * titles for each column
    if titles is not None:
        assert len(titles) == len(imgs[0])
        canvas = np.pad(canvas, [[font_size + 4, 0], [0, 0], [0, 0]], "constant")
        txts, x_acc = [], 0
        for c, title in enumerate(titles):
            x = x_acc + imgs[0][c].shape[1] // 2
            y = 0
            color = (255, 255, 255)
            if not isinstance(title, str):
                title, color = title
            txts.append(Text(txt=title, pos=(x, y), align=("center", "top"), font_size=font_size, color=color))
            x_acc += imgs[0][c].shape[1]
        canvas = put_texts(canvas, txts, font_size=font_size)

    return canvas
