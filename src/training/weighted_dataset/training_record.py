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


class TrainingRecord(DictWrapper):
    """Dictionary-like object to hold records about a training run"""
    def __init__(self, metrics=None, metadata=None, config=None, alpha=.1, **kwargs):
        super(TrainingRecord, self).__init__()

        self.update({
            "metrics": {
                "loss": []
            },
            "epochs": [],
            "step": 0,
            "loss":None,
            "best_loss":None
        })

        # Smoothing parameter for the loss running average
        self.alpha = alpha

        if metrics is not None:
            for metric in metrics:
                self["metrics"][metric] = []

        if metadata is not None:
            self["metadata"] = metadata

        if config is not None:
            self["config"] = config

        self.update(kwargs)

    @property
    def metrics(self):
        return self["metrics"]

    @property
    def losses(self):
        return self["metrics"]["loss"]

    @property
    def loss(self):
        return self["last_loss"]

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
        if self["loss"] is None:
            self["loss"] = loss
            self["best_loss"] = loss
        else:
            self["loss"] = self["loss"]*(1-self.alpha) + self.alpha*loss
            self["best_loss"] = min(self["best_loss"], self["loss"] )

    def next_step(self):
        self["step"] += 1

    def new_epoch(self):
        self["epochs"].append(self.step)

    def __repr__(self):
        step = self["step"]
        epoch = len(self["epochs"])
        metrics = [k for k in self["metrics"] if k != "loss"]
        if len(metrics) > 0:
            metric_report = ", metrics: "+str(metrics)
        else:
            metric_report = ""
        try:
            loss_report = ", loss: "+str(self.loss)
        except IndexError:
            loss_report = ""

        return f"{self.__class__.__name__}(step: {step}, epoch: {epoch}{loss_report}{metric_report})"

    def __str__(self):
        return self.__repr__()
