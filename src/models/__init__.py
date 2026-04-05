from .DiT import DiT_models, DiT
from .ShiftSynth import ShiftSynth_models, ShiftSynth
from .DriftSynth import DriftSynth_models, DriftSynth
from .MaskSynth import MaskSynth_models, MaskSynth

All_models = {**DiT_models, **ShiftSynth_models, **DriftSynth_models, **MaskSynth_models}
