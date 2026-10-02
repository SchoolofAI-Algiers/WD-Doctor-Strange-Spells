from types import SimpleNamespace

import pytest

import src.rendering.spell_renderer as sr
from src.rendering.spell_renderer import ASSET_RADIUS, SpellConfig, SpellRenderer


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.setattr(sr, "load_asset", lambda p: None)
    return SpellRenderer(SpellConfig(), (480, 640))


def geo(size=100.0, center=(320.0, 240.0), angle=0.0):
    return SimpleNamespace(hand_size_px=size, palm_center_px=center, palm_angle_rad=angle)


def gest(conf=1.0):
    return SimpleNamespace(confidence=conf)


def test_scale_gives_radius_base_scale_times_hand_size(renderer):
    tr = renderer.compute_transform(geo(size=100), gest(), None, 0.0)
    assert tr.scale * ASSET_RADIUS == pytest.approx(100 * renderer.cfg.base_scale)


def test_scale_is_linear_in_hand_size(renderer):
    s1 = renderer.compute_transform(geo(size=80), gest(), None, 0.0).scale
    s2 = renderer.compute_transform(geo(size=160), gest(), None, 0.0).scale
    assert s2 == pytest.approx(2 * s1)


def test_position_and_rotation_pass_through(renderer):
    tr = renderer.compute_transform(geo(center=(111.0, 222.0), angle=0.7), gest(), None, 0.0)
    assert tr.position_px == (111.0, 222.0)
    assert tr.rotation_rad == pytest.approx(0.7)


def test_opacity_is_config_times_confidence(renderer):
    tr = renderer.compute_transform(geo(), gest(0.5), None, 0.0)
    assert tr.opacity == pytest.approx(renderer.cfg.opacity * 0.5)


def test_zero_confidence_gives_zero_opacity(renderer):
    assert renderer.compute_transform(geo(), gest(0.0), None, 0.0).opacity == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))