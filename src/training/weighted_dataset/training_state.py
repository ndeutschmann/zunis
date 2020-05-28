from collections.abc import MutableMapping


class DictWrapper(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class TrainingState(DictWrapper):
    def __init__(self, metrics=None, metadata=None, **kwargs):
        super(TrainingState, self).__init__()

        self.update({
            "metrics": {
                "loss": []
            },
            "epochs": [],
            "step": 0,
        })

        if metrics is not None:
            for metric in metrics:
                self["metrics"][metric] = []

        if metadata is not None:
            self["metadata"] = metadata

        self.update(kwargs)

    @property
    def metrics(self):
        return self["metrics"]

    @property
    def losses(self):
        return self["metrics"]["loss"]

    @property
    def loss(self):
        return self["metrics"]["loss"][-1]

    @property
    def step(self):
        return self["step"]

    @property
    def epoch(self):
        return len(self["epochs"])

    def log_metric(self, value, metric):
        self["metrics"][metric].append(value)

    def log_loss(self, loss):
        self.log_metric(loss, "loss")

    def next_step(self):
        self["step"] += 1

    def next_epoch(self):
        self["epochs"].append(self.step)
