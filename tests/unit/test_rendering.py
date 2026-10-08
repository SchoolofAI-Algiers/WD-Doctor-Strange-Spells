"""Unit tests for spell_renderer.py.

Assets are generated in pytest's temp folder: no camera or real files needed.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from src.rendering import spell_renderer as sr

H, W = 240, 320


# ---------------------------------------------------------------- helpers
def write_png(path, img):
    assert cv2.imwrite(str(path), img)
    return str(path)


def write_disc(path, bgr=(0, 0, 255), size=65, radius=30):
    img = np.zeros((size, size, 4), np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (*bgr, 255), -1)
    return write_png(path, img)


def geom(size=30.0, center=(160.0, 120.0), angle=0.0):
    return SimpleNamespace(hand_size_px=size, palm_center_px=center, palm_angle_rad=angle)


def gest(active=True, confidence=1.0):
    state = sr.GestureState.ACTIVE if active else sr.GestureState.IDLE
    return SimpleNamespace(state=state, confidence=confidence)


def step(r, n=1, geometry=None, gesture=None, hand_id=0, spell_id="default"):
    """Call compute_transform n times and return the last transform."""
    for _ in range(n):
        t = r.compute_transform(
            geometry or geom(), gesture or gest(), None, 0.0, hand_id=hand_id, spell_id=spell_id
        )
    return t


def tr(pos=(160.0, 120.0), radius_px=45.0, rot=0.0, opacity=1.0):
    return sr.SpellTransform(pos, radius_px / sr.ASSET_RADIUS, rot, opacity, "default")


def make_renderer(path, layers=None):
    layers = layers or [sr.LayerConfig(path)]
    return sr.SpellRenderer({"default": sr.SpellConfig(layers=layers)}, (H, W))


def painted(frame):
    return int((frame.astype(int).sum(axis=2) > 0).sum())


@pytest.fixture(autouse=True)
def clear_cache():
    sr._ASSET_CACHE.clear()


@pytest.fixture
def red_disc(tmp_path):
    return write_disc(tmp_path / "red.png", (0, 0, 255))


@pytest.fixture
def green_disc(tmp_path):
    return write_disc(tmp_path / "green.png", (0, 255, 0))


@pytest.fixture
def renderer(red_disc):
    return make_renderer(red_disc)


@pytest.fixture
def frame():
    return np.zeros((H, W, 3), np.uint8)


@pytest.fixture
def line_art(tmp_path):
    """White page with four ink patches (PNG, so values are exact)."""
    img = np.full((512, 512, 3), 255, np.uint8)
    img[100:200, 100:200] = (0, 0, 255)  # red ink
    img[300:400, 100:200] = (128, 128, 128)  # mid grey
    img[100:200, 300:400] = (245, 245, 245)  # JPEG-style haze
    img[300:400, 300:400] = (0, 0, 0)  # black ink
    return write_png(tmp_path / "art.png", img)


# ------------------------------------------------------- pure functions
def test_layer_angle():
    assert sr.layer_angle(1.0, 2.0, 3.0) == pytest.approx(7.0)
    assert sr.layer_angle(0.5, 0.0, 99.0) == pytest.approx(0.5)
    assert sr.layer_angle(0.0, -1.0, 2.0) == pytest.approx(-2.0)


@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi / 2, -2.1])
@pytest.mark.parametrize("scale", [0.1, 2.5])
def test_build_affine_maps_source_centre_to_destination(angle, scale):
    M = sr.build_affine(angle, scale, (256, 256), (100.5, 42.25))
    np.testing.assert_allclose(M @ [256.0, 256.0, 1.0], (100.5, 42.25), atol=1e-9)


def test_build_affine_rotation_scale_and_out_buffer():
    np.testing.assert_allclose(
        sr.build_affine(0.0, 1.0, (0, 0), (0, 0)), [[1, 0, 0], [0, 1, 0]], atol=1e-12
    )
    out = np.zeros((2, 3))
    M = sr.build_affine(math.pi / 2, 3.0, (0, 0), (0, 0), out=out)
    assert M is out
    np.testing.assert_allclose(M @ [1, 0, 1], (0, 3), atol=1e-12)
    np.testing.assert_allclose(M @ [0, 1, 1], (-3, 0), atol=1e-12)


@pytest.mark.parametrize(
    "bg, colour, alpha, expected",
    [
        (77, (0, 0, 0), 0.0, [77, 77, 77]),  # transparent: background kept
        (77, (10, 20, 30), 1.0, [10, 20, 30]),  # opaque: replaced
        (100, (100, 0, 0), 0.5, [150, 50, 50]),  # half alpha: blended
        (10, (20, 20, 20), 0.0, [30, 30, 30]),  # premultiplied colour is added
        (200, (200, 200, 200), 0.5, [255, 255, 255]),  # clipped
    ],
)
def test_composite_premultiplied(bg, colour, alpha, expected):
    roi = np.full((2, 2, 3), bg, np.uint8)
    layer = np.zeros((2, 2, 4), np.float32)
    layer[..., :3], layer[..., 3] = colour, alpha
    tmp, inv = np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 1), np.float32)
    assert sr.composite_premultiplied(roi, layer, tmp, inv) is None
    assert (roi == np.array(expected)).all()


# ------------------------------------------------------------ load_asset
def test_load_asset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        sr.load_asset(str(tmp_path / "nope.png"))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError):
        sr.load_asset(str(bad))


def test_load_asset_rgba_is_premultiplied_512_float32(tmp_path):
    img = np.zeros((32, 32, 4), np.uint8)
    img[...] = (200, 100, 50, 128)
    a = sr.load_asset(write_png(tmp_path / "half.png", img))
    alpha = 128 / 255
    assert a.shape == (512, 512, 4) and a.dtype == np.float32
    assert a.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(a[256, 256], [200 * alpha, 100 * alpha, 50 * alpha, alpha], atol=0.5)


@pytest.mark.parametrize(
    "img, expected",
    [
        (np.full((16, 16, 3), (10, 20, 30), np.uint8), [10, 20, 30, 1.0]),  # BGR
        (np.full((16, 16), 90, np.uint8), [90, 90, 90, 1.0]),  # grayscale
    ],
)
def test_load_asset_opaque_inputs(tmp_path, img, expected):
    a = sr.load_asset(write_png(tmp_path / "img.png", img))
    np.testing.assert_allclose(a[256, 256], expected, atol=1e-3)


@pytest.mark.parametrize("h, w", [(50, 100), (100, 50), (1024, 1024)])
def test_load_asset_keeps_aspect_ratio_and_centres(tmp_path, h, w):
    img = np.zeros((h, w, 4), np.uint8)
    img[...] = (10, 20, 30, 255)
    a = sr.load_asset(write_png(tmp_path / "img.png", img))
    assert a.shape == (512, 512, 4)
    alpha = a[..., 3]
    if h < w:
        assert alpha[:128].max() == 0 and alpha[384:].max() == 0
        assert alpha[128:384].min() == pytest.approx(1.0, abs=1e-3)
    elif w < h:
        assert alpha[:, :128].max() == 0 and alpha[:, 384:].max() == 0
        assert alpha[:, 128:384].min() == pytest.approx(1.0, abs=1e-3)
    else:
        assert alpha.min() == pytest.approx(1.0, abs=1e-3)


def test_key_white_makes_paper_transparent_and_keeps_ink(line_art):
    a = sr.load_asset(line_art, key_white=True)
    np.testing.assert_allclose(a[10, 10], [0, 0, 0, 0], atol=1e-4)  # paper
    np.testing.assert_allclose(a[150, 150], [0, 0, 255, 1.0], atol=1e-3)  # red ink
    np.testing.assert_allclose(a[350, 350], [0, 0, 0, 1.0], atol=1e-3)  # black ink
    assert a[350, 150, 3] == pytest.approx(1 - 128 / 255, abs=1e-3)  # grey
    assert a[150, 350, 3] == 0.0  # haze wiped


def test_key_white_never_produces_negative_colour(line_art):
    # Negative premultiplied colour would DARKEN the camera image around faint haze.
    assert sr.load_asset(line_art, key_white=True)[..., :3].min() >= 0.0


def test_key_white_threshold_and_coloured_ink(tmp_path):
    img = np.full((512, 512, 3), 255, np.uint8)
    img[:256, :256] = 240  # faint ink (kept)
    img[:256, 256:] = 241  # haze (wiped)
    img[300:400, 100:200] = (200, 150, 100)  # lightest channel = 100
    a = sr.load_asset(write_png(tmp_path / "k.png", img), key_white=True)
    assert a[100, 100, 3] == pytest.approx(15 / 255, abs=1e-3)
    assert a[100, 400, 3] == 0.0
    np.testing.assert_allclose(a[350, 150], [100, 50, 0, 1 - 100 / 255], atol=1e-3)


def test_key_white_is_ignored_when_the_image_has_alpha(tmp_path):
    img = np.full((32, 32, 4), 255, np.uint8)  # opaque white
    a = sr.load_asset(write_png(tmp_path / "w.png", img), key_white=True)
    np.testing.assert_allclose(a[256, 256], [255, 255, 255, 1.0], atol=1e-3)


def test_tint_recolours_and_keeps_alpha(tmp_path):
    img = np.zeros((32, 32, 4), np.uint8)
    img[...] = (10, 20, 30, 128)
    a = sr.load_asset(write_png(tmp_path / "t.png", img), tint=(0, 255, 0))
    alpha = 128 / 255
    np.testing.assert_allclose(a[256, 256], [0, 255 * alpha, 0, alpha], atol=0.5)


def test_load_asset_cache(tmp_path):
    p = Path(write_disc(tmp_path / "gone.png"))
    first = sr.load_asset(str(p))
    assert sr.load_asset(str(p)) is first
    assert sr.load_asset(str(p), key_white=True) is not first
    assert sr.load_asset(str(p), tint=(1, 2, 3)) is not first
    p.unlink()
    assert sr.load_asset(str(p)) is first  # served from the cache


# --------------------------------------------------- SpellRenderer: init
def test_init_rejects_empty_spell_and_missing_asset(tmp_path):
    with pytest.raises(ValueError, match="no layers"):
        sr.SpellRenderer({"bad": sr.SpellConfig()}, (H, W))
    cfg = sr.SpellConfig(layers=[sr.LayerConfig(str(tmp_path / "x.png"))])
    with pytest.raises(FileNotFoundError):
        sr.SpellRenderer({"s": cfg}, (H, W))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, 1.0),
        ({"scale_mult": 1.1}, 1.1),
        ({"scale_mult": 1.0, "offset": (3.0, 4.0)}, 6.0),
    ],
)
def test_extent_covers_scale_and_offset(red_disc, kwargs, expected):
    r = make_renderer(red_disc, layers=[sr.LayerConfig(red_disc, **kwargs)])
    assert r._extent["default"] == pytest.approx(expected)


# ------------------------------------------------- SpellRenderer: transform
def test_unknown_spell_raises_key_error(renderer):
    with pytest.raises(KeyError):
        step(renderer, spell_id="nope")


@pytest.mark.parametrize(
    "size, radius_px",
    [
        (30.0, 45.0),  # 30 * base_scale 1.5
        (1.0, 20.0),  # clamped up to min_radius_px
        (10_000.0, 320.0),  # clamped down to the largest frame side
        (float("nan"), 30.0),  # bad measurement -> min size (20) * 1.5
        (float("inf"), 30.0),
        (0.0, 30.0),
        (-5.0, 30.0),
    ],
)
def test_scale_from_hand_size(renderer, size, radius_px):
    t = step(renderer, geometry=geom(size=size, center=(33.0, 44.0), angle=0.25))
    assert t.scale == pytest.approx(radius_px / sr.ASSET_RADIUS)
    assert (t.position_px, t.rotation_rad, t.spell_id) == ((33.0, 44.0), 0.25, "default")


def test_fade_in_then_out(renderer):
    assert step(renderer).opacity == pytest.approx(0.9 * 0.2)  # 1/fade_in_frames
    assert step(renderer, 19).opacity == pytest.approx(0.9)  # clamped at full
    assert step(renderer, gesture=gest(False)).opacity == pytest.approx(0.9 * 0.875)
    assert step(renderer, 30, gesture=gest(False)).opacity == 0.0


def test_hands_fade_independently_and_confidence_scales_opacity(renderer):
    assert step(renderer, 3, hand_id=0).opacity == pytest.approx(0.9 * 0.6)
    assert step(renderer, 1, hand_id=1).opacity == pytest.approx(0.9 * 0.2)
    assert step(renderer, 3, gesture=gest(False), hand_id=2).opacity == 0.0
    assert step(renderer, 5, gesture=gest(True, 0.5), hand_id=3).opacity == pytest.approx(0.45)


def test_switching_spell_restarts_the_fade(red_disc):
    cfg = sr.SpellConfig(layers=[sr.LayerConfig(red_disc)])
    r = sr.SpellRenderer({"a": cfg, "b": cfg}, (H, W))
    step(r, 5, spell_id="a")
    t = step(r, 1, spell_id="b")
    assert t.spell_id == "b" and t.opacity == pytest.approx(0.9 * 0.2)


# ------------------------------------------------- SpellRenderer: ghosts
def test_lost_hand_fades_as_a_ghost_then_is_forgotten(renderer):
    t = step(renderer, 5)
    ((ghost, no_gesture),) = renderer.ghost_spells(set())
    assert ghost is not t and no_gesture is None
    assert t.opacity == pytest.approx(0.9)  # original not mutated
    assert ghost.opacity == pytest.approx(0.9 * 0.875)
    for _ in range(6):
        assert len(renderer.ghost_spells(set())) == 1
    assert renderer.ghost_spells(set()) == []  # fade_out_frames (8) reached
    assert 0 not in renderer._hands


def test_ghost_spells_keeps_seen_hands_and_drops_invisible_ones(renderer):
    step(renderer, 5, hand_id=0)
    step(renderer, 3, gesture=gest(False), hand_id=1)  # never became visible
    renderer._hands[5] = sr._HandState()  # no transform / spell yet
    assert renderer.ghost_spells({0}) == []
    assert set(renderer._hands) == {0}


# ----------------------------------------------------- SpellRenderer: render
@pytest.mark.parametrize(
    "t",
    [
        tr(opacity=0.0),
        tr(opacity=-1.0),
        tr(pos=(-500.0, 120.0)),
        tr(pos=(900.0, 120.0)),
        tr(pos=(160.0, -500.0)),
        tr(pos=(160.0, 700.0)),
    ],
    ids=["zero-opacity", "negative-opacity", "left", "right", "top", "bottom"],
)
def test_nothing_is_drawn(renderer, frame, t):
    renderer.render(frame, t, None, 0.0)
    assert painted(frame) == 0


def test_paints_a_disc_and_returns_the_same_frame(renderer, frame):
    assert renderer.render(frame, tr(), None, 0.0) is frame
    np.testing.assert_allclose(frame[120, 160], [0, 0, 255], atol=2)
    assert (frame[120, 220] == 0).all() and (frame[0, 0] == 0).all()


def test_opacity_blends_with_background_and_is_clamped(renderer, frame):
    bg = np.full((H, W, 3), 100, np.uint8)
    renderer.render(bg, tr(opacity=0.5), None, 0.0)
    np.testing.assert_allclose(bg[120, 160], [50, 50, 177.5], atol=2)
    assert (bg[0, 0] == 100).all()
    renderer.render(frame, tr(opacity=5.0), None, 0.0)
    np.testing.assert_allclose(frame[120, 160], [0, 0, 255], atol=2)


@pytest.mark.parametrize(
    "pos, hit, miss",
    [
        ((0.0, 0.0), (0, 0), (W - 1, H - 1)),
        ((W - 1.0, H - 1.0), (W - 1, H - 1), (0, 0)),
    ],
)
def test_spell_is_clipped_at_frame_corners(renderer, frame, pos, hit, miss):
    renderer.render(frame, tr(pos=pos), None, 0.0)
    assert frame[hit[1], hit[0], 2] > 200
    assert (frame[miss[1], miss[0]] == 0).all()


@pytest.mark.parametrize(
    "angle, centre, empty",
    [
        (0.0, (145, 120), (55, 120)),  # offset (1, 0): one radius to the right
        (math.pi / 2, (100, 165), (100, 75)),  # hand turned 90 deg: "right" becomes "down"
    ],
)
def test_layer_offset_rotates_with_the_hand(red_disc, frame, angle, centre, empty):
    r = make_renderer(red_disc, layers=[sr.LayerConfig(red_disc, offset=(1.0, 0.0))])
    r.render(frame, tr(pos=(100.0, 120.0), rot=angle), None, 0.0)
    assert frame[centre[1], centre[0], 2] > 200
    assert (frame[empty[1], empty[0]] == 0).all()


def test_scale_mult_shrinks_the_layer(red_disc):
    full, half = np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)
    make_renderer(red_disc).render(full, tr(), None, 0.0)
    make_renderer(red_disc, layers=[sr.LayerConfig(red_disc, scale_mult=0.5)]).render(
        half, tr(), None, 0.0
    )
    assert painted(half) / painted(full) == pytest.approx(0.25, abs=0.04)  # area ~ radius^2


@pytest.mark.parametrize("green_on_top, expected", [(True, [0, 255, 0]), (False, [0, 0, 255])])
def test_later_layers_are_drawn_on_top(red_disc, green_disc, frame, green_on_top, expected):
    a, b = (red_disc, green_disc) if green_on_top else (green_disc, red_disc)
    r = make_renderer(red_disc, layers=[sr.LayerConfig(a), sr.LayerConfig(b)])
    r.render(frame, tr(), None, 0.0)
    np.testing.assert_allclose(frame[120, 160], [*expected], atol=2)


def test_layer_opacity_mult(red_disc, frame):
    r = make_renderer(red_disc, layers=[sr.LayerConfig(red_disc, opacity_mult=0.5)])
    r.render(frame, tr(), None, 0.0)
    assert frame[120, 160, 2] == pytest.approx(127.5, abs=2)


def test_angle_and_scale_passed_to_build_affine(red_disc, green_disc, frame, monkeypatch):
    calls, real = [], sr.build_affine

    def spy(angle, scale, src, dst, out=None):
        calls.append((angle, scale))
        return real(angle, scale, src, dst, out=out)

    monkeypatch.setattr(sr, "build_affine", spy)
    layers = [
        sr.LayerConfig(red_disc, rotation_speed=2.0),
        sr.LayerConfig(green_disc, rotation_speed=2.0, follow_hand=False, scale_mult=0.5),
    ]
    t = tr(rot=0.7)
    make_renderer(red_disc, layers=layers).render(frame, t, None, 1.5)
    assert calls[0] == pytest.approx((0.7 + 3.0, t.scale))  # follows the hand + spin
    assert calls[1] == pytest.approx((3.0, t.scale * 0.5))  # upright + spin


def test_shared_scratch_buffers_do_not_leak_between_windows(red_disc):
    big, small = tr(pos=(100.0, 120.0)), tr(pos=(250.0, 120.0), radius_px=20.0)
    both = np.zeros((H, W, 3), np.uint8)
    r = make_renderer(red_disc)
    r.render(both, big, None, 0.0)
    r.render(both, small, None, 0.0)
    alone = np.zeros((H, W, 3), np.uint8)
    make_renderer(red_disc).render(alone, small, None, 0.0)
    np.testing.assert_array_equal(both[:, 200:], alone[:, 200:])


def test_render_all_draws_every_entry_including_ghosts(renderer, frame):
    assert renderer.render_all(frame, [], 0.0) is frame and painted(frame) == 0
    spells = [(tr(pos=(80.0, 120.0)), gest()), (tr(pos=(240.0, 120.0)), None)]
    renderer.render_all(frame, spells, 0.0)
    assert frame[120, 80, 2] > 200 and frame[120, 240, 2] > 200
    assert (frame[120, 160] == 0).all()


def test_hand_lifecycle_appear_lose_fade_disappear(renderer):
    t = step(renderer, 5)
    shown = np.zeros((H, W, 3), np.uint8)
    renderer.render_all(shown, [(t, gest())], 0.0)
    assert painted(shown) > 0
    for _ in range(7):  # tracker lost the hand
        f = np.zeros((H, W, 3), np.uint8)
        renderer.render_all(f, renderer.ghost_spells(set()), 0.0)
        assert painted(f) > 0
    f = np.zeros((H, W, 3), np.uint8)
    renderer.render_all(f, renderer.ghost_spells(set()), 0.0)
    assert painted(f) == 0
