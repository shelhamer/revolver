from torch.utils.data.dataloader import default_collate


class InputsTargetAuxCollate(object):
    """
    Custom collate to check the no. of elements, insist that the last element
    is a dict (for auxiliary information), and pass the auxiliary unchanged.
    """

    def __call__(self, batch):
        batch = batch[0]  # assume batch size one
        if len(batch) >= 3 and isinstance(batch[-1], dict):
            inputs, target, aux = batch[:-2], batch[-2], batch[-1]
            inputs = list(inputs)
            for i, input_ in enumerate(inputs):
                if not isinstance(input_, list):
                    inputs[i] = default_collate(input_).unsqueeze(0)
                else:
                    inputs[i] = [[default_collate(in_).unsqueeze(0)
                                  for in_ in inp] for inp in input_]
            target = default_collate(target)
            return (*inputs, target, aux)
        raise TypeError("Data should contain (inputs, target, aux) tuples; "
                        "found: {}".format(type(batch)))


class Wrapper(object):
    """
    Mixin for deferring attributes to a wrapped, inner object.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, attr):
        """
        Dispatch attributes by their status as magic, members, or missing.

        - magic is handled by the standard getattr
        - existing attributes are returned
        - missing attributes are deferred to the inner object.
        """
        # don't make magic any more magical
        is_magic = attr.startswith('__') and attr.endswith('__')
        if is_magic:
            return super().__getattr__(attr)
        try:
            # try to return the attribute...
            return self.__dict__[attr]
        except:
            # ...and defer to the inner dataset if it's not here
            return getattr(self.inner, attr)
