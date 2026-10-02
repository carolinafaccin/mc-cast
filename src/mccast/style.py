"""Figure style: the brand (brand.py, synced from the lina-brand repository) plus this
repository's source line and repository name."""
from . import brand
from .brand import *  # noqa: F401,F403  colors, data palettes, helpers
from .config import ROOT

SOURCE = "Source: MapBiomas Collection 10; OpenStreetMap; own model (MC-CAST)."
REPO = "github.com/carolinafaccin/mc-cast"

# Model classes (ids in config.yaml) -> colors, as in the coastal repository's land-cover maps
CLASS_COLORS = {0: "white", 1: SAGE_DARK, 2: PEACH, 3: ORANGE, 4: MAP_WATER, 5: OTHER}


def setup():
    """Register the bundled fonts (OFL) and set the matplotlib defaults."""
    brand.setup(ROOT / "assets" / "fonts")


def footer(fig, note=SOURCE):
    brand.footer(fig, note, REPO)
