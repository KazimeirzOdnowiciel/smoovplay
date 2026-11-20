"""An HDR desktop's pixels as the SDR picture they show (framegen/scrgb.py)."""
import numpy as np
import pytest

from framegen import scrgb


def _scrgb(level, white):
    """An SDR level (0-255) as an HDR desktop holds it: linear, SDR white at `white`."""
    v = np.asarray(level, np.float64) / 255.0
    return white * np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


@pytest.mark.parametrize("white", [1.0, 2.5, 5.0])
def test_an_sdr_picture_on_an_hdr_desktop_comes_back_as_it_was(white):
    """ddagrab's 8-bit capture gave a test pattern's 64 as 33 and everything from 192 up as 255."""
    levels = np.arange(256)
    rgba = np.ones((8, 256, 4), np.float16)
    rgba[..., 0] = _scrgb(levels, white)  # red: the ramp
    rgba[..., 1] = _scrgb(255 - levels, white)  # green: the ramp backwards
    rgba[..., 2] = _scrgb(128, white)
    bgr = scrgb.to_bgr(rgba, white, (256, 8))
    assert bgr.shape == (8, 256, 3) and bgr.dtype == np.uint8
    assert np.abs(bgr[4, :, 2].astype(int) - levels).max() <= 1
    assert np.abs(bgr[4, :, 1].astype(int) - (255 - levels)).max() <= 1
    assert np.abs(bgr[4, :, 0].astype(int) - 128).max() <= 1


def test_every_float16_value_is_within_a_level_of_srgb():
    white = 2.5
    vals = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    vals = np.unique(vals[np.isfinite(vals) & (vals >= 0) & (vals <= white * 1.5)])
    rgba = np.zeros((1, len(vals), 4), np.float16)
    rgba[0, :, 0] = vals
    got = scrgb.to_bgr(rgba, white, (len(vals), 1))[0, :, 2].astype(int)
    want = np.round(scrgb.srgb(vals.astype(np.float64) / white) * 255).astype(int)
    assert np.abs(got - want).max() <= 1


def test_it_scales_to_the_streams_size_and_cuts_off_above_sdr_white():
    rgba = np.full((36, 64, 4), 2.5 * 3, np.float16)  # an HDR app's highlights, above SDR white
    rgba[:18] = _scrgb(100, 2.5)
    bgr = scrgb.to_bgr(rgba, 2.5, (32, 18))
    assert bgr.shape == (18, 32, 3)
    assert (bgr[-2:] == 255).all()
    assert np.abs(bgr[:6].astype(int) - 100).max() <= 1
