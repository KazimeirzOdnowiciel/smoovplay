"""An HDR desktop's pixels as the SDR picture they show, for window capture with Windows' HDR on (framegen/window.py).

With Windows' HDR on, the desktop is composed in scRGB: linear light, 1.0 = 80 nits, SDR content's white at Windows'
SDR white level (Settings' "SDR content brightness"; dxgi_present.sdr_white). ddagrab's 8-bit capture of such a
desktop hands on those linear values cut off at 1.0, without sRGB's curve: a test pattern's grey 64 came back 33, 128
came back 138, and everything from 192 up came back 255, so an app's picture arrived with crushed shadows, blown
highlights and strong colours. So window.py captures the desktop as 16-bit floats instead, and `to_bgr` turns them
into the SDR picture the app drew: sRGB, SDR white as 255, at the stream's size. An SDR app (the Xbox app streams in
SDR) tops out at exactly SDR white, so nothing of it is lost; an HDR app's brighter highlights are cut off at SDR
white, and colours outside sRGB's (negative scRGB values) are taken as their size.

How: OpenCV turns float16 into 8 bits only linearly (convertScaleAbs), and 8 linear bits are too few for sRGB's
shadows, so it makes two: a fine one for the darkest SPLIT/255 of the range, and a coarse one for the rest. A table
each turns them into sRGB, and the larger of the two is the pixel: within 1 level of sRGB for every float16 value.
About 10 ms for 1902x1072 on a Ryzen 7 7800X3D, on OpenCV's own threads, which leave Python's lock and torch's
threads to the video loop. (Tried first: torch on the CPU, as fast, but its threads are the ones the video loop's
repeat check uses, and the loop's ticks went from 16 to 27 ms; DirectML, 7 ms, but running it beside RIFE's
DirectML from another thread made the driver drop the card: "the GPU device instance has been suspended".)
"""
from __future__ import annotations

import numpy as np

SPLIT = 20  # coarse levels below this come from the fine image
THREADS = 4  # OpenCV's threads: all 16 in bursts competed with the video loop for the cores
_tables = None


def srgb(x: np.ndarray) -> np.ndarray:
    """sRGB's curve: linear 0..1 -> encoded 0..1."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _make_tables() -> tuple[np.ndarray, np.ndarray]:
    levels = np.arange(256)
    fine = np.round(srgb(levels * SPLIT / 255 / 255) * 255).astype(np.uint8)
    fine[255] = 0  # beyond the fine range: the coarse image has it
    coarse = np.round(srgb(levels / 255) * 255).astype(np.uint8)
    coarse[:SPLIT] = 0  # within the fine range: the fine image has it
    return fine, coarse


def to_bgr(rgba: np.ndarray, white: float, size: tuple[int, int]) -> np.ndarray:
    """rgba: HxWx4 float16 scRGB (as ddagrab's 16-bit capture gives it); white: Windows' SDR white level in scRGB;
    size: (w, h) wanted. Returns hxwx3 uint8 BGR in sRGB."""
    global _tables
    import cv2
    if _tables is None:
        _tables = _make_tables()
        if cv2.getNumThreads() > THREADS:
            cv2.setNumThreads(THREADS)
    fine =cv2.convertScaleAbs(rgba, alpha=255.0 * 255.0 / (SPLIT * white))
    coarse = cv2.convertScaleAbs(rgba, alpha=255.0 / white)
    out = cv2.max(cv2.LUT(fine, _tables[0]), cv2.LUT(coarse, _tables[1]))
    out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)
    if (out.shape[1], out.shape[0]) != tuple(size):
        out = cv2.resize(out, tuple(size), interpolation=cv2.INTER_LINEAR)
    return out
