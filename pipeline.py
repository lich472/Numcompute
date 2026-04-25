import numpy as np

__all__ = ["Pipeline"]


def require_transformer(name, obj):
    missing = [m for m in ("fit", "transform")
               if not callable(getattr(obj, m, None))]
    if missing:
        raise TypeError(
            f"Step '{name}' is missing {missing}. Transformers need fit(X) and transform(X).")


def is_estimator(obj):
    return callable(getattr(obj, "predict", None))


class Pipeline:

    def __init__(self, steps):
        self.validate(steps)
        self.steps = steps
        self.fitted = False

    def validate(self, steps):
        if not steps:
            raise ValueError("Pipeline needs at least one step.")
        seen = set()
        for i, item in enumerate(steps):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(
                    f"Each step should be a (name, object) tuple, got {item!r} at index {i}.")
            name, obj = item
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"Step names must be non-empty strings, got {name!r}.")
            if name in seen:
                raise ValueError(
                    f"Duplicate step name '{name}' — all step names must be unique.")
            seen.add(name)
            if i < len(steps) - 1:
                require_transformer(name, obj)
        last_name, last_obj = steps[-1]
        if not is_estimator(last_obj):
            require_transformer(last_name, last_obj)

    @property
    def named_steps(self):
        return dict(self.steps)

    def __getitem__(self, name):
        return self.named_steps[name]

    def middle_steps(self):
        return self.steps[:-1]

    def last(self):
        return self.steps[-1]

    def fit(self, X, y=None):
        X_cur = X
        for name, step in self.middle_steps():
            X_cur = step.fit_transform(X_cur)
        last_name, last_step = self.last()
        if is_estimator(last_step):
            if y is None:
                raise ValueError(
                    f"Step '{last_name}' is a model and needs y to train on.")
            last_step.fit(X_cur, y)
        else:
            last_step.fit(X_cur)
        self.fitted = True
        return self

    def fit_transform(self, X, y=None):
        if is_estimator(self.last()[1]):
            raise TypeError(
                "Last step is a model — use fit() then predict() instead.")
        X_cur = X
        for _, step in self.middle_steps():
            X_cur = step.fit_transform(X_cur)
        self.fitted = True
        return self.last()[1].fit_transform(X_cur)

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "Call fit() or fit_transform() before transform().")
        if is_estimator(self.last()[1]):
            raise TypeError("Last step is a model — use predict() instead.")
        X_cur = X
        for _, step in self.steps:
            X_cur = step.transform(X_cur)
        return X_cur

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        if not is_estimator(self.last()[1]):
            raise TypeError(
                "Last step isn't a model — use transform() instead.")
        X_cur = X
        for _, step in self.middle_steps():
            X_cur = step.transform(X_cur)
        return self.last()[1].predict(X_cur)

    def set_params(self, **params):
        named = self.named_steps
        for key, val in params.items():
            if "__" not in key:
                raise ValueError(
                    f"Keys must be 'step_name__param_name', got '{key}'.")
            step_name, param = key.split("__", 1)
            if step_name not in named:
                raise ValueError(
                    f"No step called '{step_name}'. Available: {list(named.keys())}.")
            if not hasattr(named[step_name], param):
                raise ValueError(
                    f"Step '{step_name}' has no attribute '{param}'.")
            setattr(named[step_name], param, val)
        return self
