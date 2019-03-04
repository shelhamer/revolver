from .fcn32s_lite import fcn32s_lite


class fgbg_lite(fcn32s_lite):
    """
    fg-bg-lite: FCN-32s-lite with a dummy arg for compatibility with
    conditional models (for instance to do fair evaluation).
    """

    def forward(self, x, *args):
        # *args will absorb the support (and other args, if any),
        # making this model compatibible with conditional models
        # in order to evaluate it on the same mask data.
        return super().forward(x)
