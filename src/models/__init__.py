from .DiT import DiT_models, DiT
from .Transformer import Transformer_models, Transformer
from .AR_DiT import AR_DiT_models, AR_DiT
from .MaskedAR import MaskedAR_models, MaskedARTransformer
All_models = {**DiT_models, **Transformer_models, **AR_DiT_models, **MaskedAR_models}
