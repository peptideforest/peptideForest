class PFParam:
    def __init__(self, value=None, strategy=None):
        """Note that the current value is not stored in history!"""
        self._value = value
        self.strategy = strategy
        self.history = []

    def __repr__(self):
        return f"PFParam(value={self.value}, strategy={self.strategy})"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.history.append(self._value)
        self._value = value


class PFConfig:
    """
    # todo: load from json
    # todo: load from dict
    """
    def __init__(self):
        self.q_cut = PFParam()
        self.n_iterations = PFParam()
        self.n_spectra = PFParam()
        self.n_folds = PFParam()
        self.n_trees = PFParam()
        self.max_depth = PFParam()
        self.learning_rate = PFParam()
        self.alpha = PFParam()
        self.lambda_ = PFParam()
        self.gamma = PFParam()
        self.engine_rescore = PFParam()

    def __next__(self):
        for name, param in vars(self).items():
            if isinstance(param, PFParam):
                param.value = self.parse_strategy(param.strategy, param.value)
        return self

    def parse_strategy(self, strategy: str, value):
        """Returns next params according to strategy either directly or as list for
        grid search.

        Possible strategies:
        - *=, +=, -=, /= => multiply, add, subtract, divide current value
        - [*=, _, *=, ...] => grid search, _ represents current value
        """
        if strategy is None:
            return value
        elif strategy.startswith("["):
            return self.parse_grid_search(strategy, value)
        elif (
            strategy.startswith("*=")
            or strategy.startswith("/=")
            or strategy.startswith("+=")
            or strategy.startswith("-=")
        ):
            return self.parse_param_change(strategy, value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def parse_grid_search(self, strategy: str, value):
        strategy = strategy.strip()
        if not strategy.startswith("[") or not strategy.endswith("]"):
            raise ValueError(f"Invalid strategy: {strategy}")
        strategy = strategy.replace("[", "").replace("]", "")
        strategy = strategy.split(",")
        strategy = [s.strip() for s in strategy]
        strategy = [self.parse_param_change(s, value) for s in strategy]
        return strategy

    @staticmethod
    def parse_param_change(strategy: str, value):
        strategy = strategy.strip()
        if strategy == "_":
            return value
        try:
            operator, change = strategy.split("=")
        except ValueError:
            raise ValueError(f"Invalid strategy: {strategy}")
        return eval(str(value) + operator + change)
