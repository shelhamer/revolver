from .fcn32s import fcn32s


class fgbg(fcn32s):
    """
    fg-bg: FCN-32s with a dummy arg for compatibility with
    conditional models (for instance to do fair evaluation).
    """

    def forward(self, x, *args):
        # *args will absorb the support (and other args, if any),
        # making this model compatibible with conditional models
        # in order to evaluate it on the same mask data.
        return super().forward(x)
