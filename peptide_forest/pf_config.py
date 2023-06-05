from copy import deepcopy, copy
# todo: sanity check value types. e.g. n_estimators should be int => strategy updating
#  with *= should result in an int not float in any case


class PFParam:
    def __init__(self, value=None, strategy=None):
        """Note that the current value is not stored in history!"""
        self._value = value
        self.strategy = strategy
        self.history = None
        self._grid = None
        self.grid_history = None

    def __repr__(self):
        return (
            f"PFParam(\n \t value={self.value}, strategy={self.strategy}, grid={self.grid},"
            f" \n \t history={self.history}, grid_history={self.grid_history} \n ) \n"
        )

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.history is None:
            self.history = [self._value]
        else:
            self.history.append(self._value)
        self._value = value
        if self.grid is None:
            self.grid = [value]

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if self.grid_history is None:
            self.grid_history = [self._grid]
        else:
            self.grid_history.append(self._grid)
        self._grid = grid


class PFConfig:
    """
    # todo: load from json
    # todo: load from dict
    # todo: add to pandas or to dict method
    """

    def __init__(self, config=None):
        self.q_cut = PFParam()
        self.n_train = PFParam()
        self.n_spectra = PFParam()
        self.n_folds = PFParam()
        self.n_estimators = PFParam()
        self.max_depth = PFParam()
        self.learning_rate = PFParam()
        self.reg_alpha = PFParam()
        self.reg_lambda = PFParam()
        self.min_split_loss = PFParam()
        self.engine_rescore = PFParam()
        self.n_jobs = PFParam()
        self.xgb_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "reg_alpha",
            "reg_lambda",
            "min_split_loss",
            "n_jobs",
        ]
        if config is not None:
            self._load(config, mode="dict")

    def __repr__(self):
        return f"PFConfig(\n {vars(self)} \n)"

    def __next__(self):
        for name, param in vars(self).items():
            if isinstance(param, PFParam):
                parsed_strategy = self.parse_strategy(param.strategy, param.value)
                if type(parsed_strategy) is list:
                    param.grid = parsed_strategy
                else:
                    param.value = parsed_strategy
                    param.grid = [parsed_strategy]
        return self

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def parse_strategy(self, strategy: str, value):
        """Returns next params according to strategy either directly or as list for
        grid search.

        Possible strategies:
        - *=, +=, -=, /= => multiply, add, subtract, divide current value
        - [*=, _, *=, ...] => grid search, _ represents current value
        """
        if strategy is None or strategy == "_":
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

    def grid(self):
        return {
            name: param.grid
            for name, param in vars(self).items()
            if name in self.xgb_params
            and param.grid is not None
            and len(param.grid) > 1
        }

    def param_dict(self):
        return {
            name: param.value
            for name, param in vars(self).items()
            if name in self.xgb_params and param.value is not None
        }

    def update_values(self, values):
        for name, value in values.items():
            if name in self.xgb_params:
                param = getattr(self, name)
                param.value = value
                param.grid = [value]
            else:
                # raise ValueError(f"Unknown parameter: {name}")
                # todo: do not let that pass silently
                pass

    def _load(self, config, mode="dict"):
        """Loads config from dict or json.

        Dict Format:
        { "name": { "value": value, "strategy": strategy_str , "grid": grid} }
        """
        if mode == "dict":
            for name, initial_config in config.items():
                if name in vars(self):
                    param = getattr(self, name)
                    param.grid = initial_config.get("grid", param.grid)
                    param.value = initial_config.get("value", param.value)
                    param.strategy = initial_config.get("strategy", param.strategy)
                    param.history = None
                    param.grid_history = None
                else:
                    raise ValueError(f"Unknown parameter: {name}")
        elif mode == "json":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown config type: {type}")
