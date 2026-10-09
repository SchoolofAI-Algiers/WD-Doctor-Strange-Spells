"""Shield spell preset for the SpellRenderer.

Inner + outer rings spin in opposite directions.
"""

from pathlib import Path

from src.rendering.spell_renderer import LayerConfig, SpellConfig

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "spells"


def make_spells() -> dict[str, SpellConfig]:
    """Return the minimal spell book: just a shield."""
    return {
        "shield": SpellConfig(
            layers=[
                LayerConfig(
                    asset=str(ASSETS / "outer" / "dark_red.png"),
                    rotation_speed=0.8,
                    scale_mult=1.0,
                ),
                LayerConfig(
                    asset=str(ASSETS / "inner" / "orange.png"),
                    rotation_speed=-1.4,
                    scale_mult=0.65,
                ),
            ],
            base_scale=1.5,
            opacity=0.9,
        )
    }
