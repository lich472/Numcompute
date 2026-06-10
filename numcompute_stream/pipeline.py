__all__ = ["Pipeline"]


def require_transformer(name, obj):
    """
    Validate that an object follows the transformer API.
    """
    missing = [
        method for method in ("fit", "transform")
        if not callable(getattr(obj, method, None))
    ]

    if missing:
        raise TypeError(
            f"Step '{name}' is missing {missing}. "
            "Transformers need fit(X) and transform(X)."
        )


def is_estimator(obj):
    """
    Check whether an object behaves like an estimator.
    """
    return callable(getattr(obj, "predict", None))


def _fit_transform_step(step, X):
    """
    Fit and transform a pipeline step.

    Uses fit_transform if available; otherwise falls back to fit().transform().
    """
    if callable(getattr(step, "fit_transform", None)):
        return step.fit_transform(X)

    return step.fit(X).transform(X)


class Pipeline:
    """
    Chain transformers and an optional final estimator.

    Parameters
    ----------
    steps : list of tuple[str, object]
        Ordered list of pipeline steps. Each step must be a
        (name, object) pair. Intermediate steps must implement
        fit(X) and transform(X). Final step can be either a transformer
        or estimator with predict(X).
    """

    def __init__(self, steps):
        self.validate(steps)
        self.steps = steps
        self.fitted = False

    def validate(self, steps):
        """
        Validate pipeline steps.
        """
        if not steps:
            raise ValueError("Pipeline needs at least one step.")

        seen = set()

        for i, item in enumerate(steps):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(
                    f"Each step should be a (name, object) tuple, "
                    f"got {item!r} at index {i}."
                )

            name, obj = item

            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"Step names must be non-empty strings, got {name!r}."
                )

            if name in seen:
                raise ValueError(
                    f"Duplicate step name '{name}' — all step names must be unique."
                )

            seen.add(name)

            if i < len(steps) - 1:
                require_transformer(name, obj)

        last_name, last_obj = steps[-1]

        if not is_estimator(last_obj):
            require_transformer(last_name, last_obj)

    @property
    def named_steps(self):
        """
        Return pipeline steps as a dictionary.
        """
        return dict(self.steps)

    def __getitem__(self, name):
        """
        Access a step by name.
        """
        return self.named_steps[name]

    def middle_steps(self):
        """
        Return all steps except the final step.
        """
        return self.steps[:-1]

    def last(self):
        """
        Return the final pipeline step.
        """
        return self.steps[-1]

    def fit(self, X, y=None):
        """
        Fit all steps in the pipeline.

        If the last step is an estimator, y is required.
        """
        X_cur = X

        for _, step in self.middle_steps():
            X_cur = _fit_transform_step(step, X_cur)

        last_name, last_step = self.last()

        if is_estimator(last_step):
            if y is None:
                raise ValueError(
                    f"Step '{last_name}' is a model and needs y to train on."
                )
            last_step.fit(X_cur, y)
        else:
            last_step.fit(X_cur)

        self.fitted = True
        return self

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data through a transformer-only pipeline.
        """
        if is_estimator(self.last()[1]):
            raise TypeError("Last step is a model — use fit() then predict() instead.")

        X_cur = X

        for _, step in self.middle_steps():
            X_cur = _fit_transform_step(step, X_cur)

        self.fitted = True
        return _fit_transform_step(self.last()[1], X_cur)

    def transform(self, X):
        """
        Transform data through a fitted transformer-only pipeline.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        if is_estimator(self.last()[1]):
            raise TypeError("Last step is a model — use predict() instead.")

        X_cur = X

        for _, step in self.steps:
            X_cur = step.transform(X_cur)

        return X_cur

    def predict(self, X):
        """
        Predict using a fitted pipeline with a final estimator.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        if not is_estimator(self.last()[1]):
            raise TypeError("Last step isn't a model — use transform() instead.")

        X_cur = X

        for _, step in self.middle_steps():
            X_cur = step.transform(X_cur)

        return self.last()[1].predict(X_cur)

    def set_params(self, **params):
        """
        Set parameters on named pipeline steps.

        Parameters must follow step_name__param_name format.
        """
        named = self.named_steps

        for key, value in params.items():
            if "__" not in key:
                raise ValueError(f"Keys must be 'step_name__param_name', got '{key}'.")

            step_name, param = key.split("__", 1)

            if step_name not in named:
                raise ValueError(
                    f"No step called '{step_name}'. Available: {list(named.keys())}."
                )

            if not hasattr(named[step_name], param):
                raise ValueError(f"Step '{step_name}' has no attribute '{param}'.")

            setattr(named[step_name], param, value)

        return self