import torch

from .fcn32s import fcn32s
from .fcn32s_lite import fcn32s_lite
from .fgbg import fgbg
from .fgbg_lite import fgbg_lite
from .interactive_early import interactive_early
from .interactive_early_lite import interactive_early_lite
from .interactive_late import interactive_late
from .interactive_late_glob import interactive_late_glob
from .interactive_late_glob_lite import interactive_late_glob_lite
from .cofeat_early import cofeat_early
from .cofeat_late import cofeat_late
from .cofeat_late_lite import cofeat_late_lite
from .cofeat_late_lite_unshared import cofeat_late_lite_unshared


models = {
    'fcn32s': fcn32s,
    'fcn32s-lite': fcn32s_lite,
    'fgbg': fgbg,
    'fgbg-lite': fgbg_lite,
    'interactive-early': interactive_early,
    'interactive-early-lite': interactive_early_lite,
    'interactive-late': interactive_late,
    'interactive-late-glob': interactive_late_glob,
    'interactive-late-glob-lite': interactive_late_glob_lite,
    'cofeat-early': cofeat_early,
    'cofeat-late': cofeat_late,
    'cofeat-late-lite': cofeat_late_lite,
    'cofeat-late-lite-unshared': cofeat_late_lite_unshared,
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
