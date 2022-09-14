from .ECG_DETR import build

def build_model(in_chan, d_model, num_class, num_queries, aux_loss=True):
    return build(in_chan, d_model, num_class, num_queries, aux_loss=aux_loss)