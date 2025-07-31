from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningOpenX
from genie.model import DINO_LAM

cli = LightningCLI(
    DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
)
