import torch

from .fcn32s import fcn32s
from .fcn32s_lite import fcn32s_lite
from .fgbg import fgbg
from .fgbg_lite import fgbg_lite
from .dios_early import dios_early
from .dios_early_lite import dios_early_lite
from .dios_late import dios_late
from .dios_late_glob import dios_late_glob
from .dios_late_glob_lite import dios_late_glob_lite
from .cofeat_early import cofeat_early
from .cofeat_late import cofeat_late
from .cofeat_late_lite import cofeat_late_lite


models = {
    'fcn32s': fcn32s,
    'fcn32s-lite': fcn32s_lite,
    'fgbg': fgbg,
    'fgbg-lite': fgbg_lite,
    'dios-early': dios_early,
    'dios-early-lite': dios_early_lite,
    'dios-late': dios_late,
    'dios-late-glob': dios_late_glob,
    'dios-late-glob-lite': dios_late_glob_lite,
    'cofeat-early': cofeat_early,
    'cofeat-late': cofeat_late,
    'cofeat-late-lite': cofeat_late_lite,
}

def prepare_model(model_name, num_classes, weights=None):
    model = models[model_name](num_classes)
    # load snapshot
    if weights:
        state = torch.load(weights)
        # include guidance for guided model from unguided snapshot
        if hasattr(model, 'z') and 'z' not in state:
            state['z'] = model.z
            state['num_z'] = model.num_z
        # override decoder for {back,for}ward compatibility
        if 'decoder.weight' in state:
            state['decoder.weight'] = model.decoder.weight
        model.load_state_dict(state)
    return model
