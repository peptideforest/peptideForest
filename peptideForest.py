# Peptide Forest 2.0.0
import json
import multiprocessing
import time


def main(
        path_dict="config/path_dict.json",
        output_file=None,
        classifier="RF-reg",
        n_train=5,
        n_eval=5,
        hyper_param_dict="config/default_hyper_parameters.json",
        train_top_data=True,
        use_cross_validation=True,
        plot_dir="./plots/",
        plot_prefix="",
        initial_engine="msgfplus",
):
    """
    Extract features from training set, impute missing values, fit model and make prediction.

    Args:
        path_dict (str): path to ursgal path dict as .json
        output_file (str): path to save new data frame to, do not save if None (default)
        classifier (str, optional): name of the classifier
        n_train (int, optional): number of training iterations
        n_eval (int, optional): number of evaluation iterations
        hyper_param_dict (dict, optional): custom hyper parameters for random forest
        train_top_data (bool, optional): only use top data (0.1% q-value) for training
        use_cross_validation (bool, optional): use of cross-validation
        plot_dir (str, optional): path to save plots to
        plot_prefix (str,optional): filename prefix for generated plots
        initial_engine (str, optional): initial engine

    """

    timer = PFTimer()
    timer["total_run_time"]

    # Import hyper parameter dict from .json file if not given otherwise
    if hyper_param_dict is None:
        with open("config/default_hyper_parameters.json") as json_dict:
            default_hyper_parameters = json.load(json_dict)
        json_dict.close()
        default_hyper_parameters["RF"]["n_jobs"] = multiprocessing.cpu_count() - 1
        default_hyper_parameters["RF-reg"]["n_jobs"] = multiprocessing.cpu_count() - 1

    print("Complete run time: {total_run_time} min".format(**timer))


class PFTimer(object):
    def __init__(self):
        self.times = {}
        self.was_stopped = set()

    def keys(self):
        return self.times.keys()

    def __getitem__(self, key):
        if key not in self.was_stopped:
            if key in self.times.keys():
                self.times[key] = round((time.time() - self.times[key]) / 60, 3)
                self.was_stopped.add(key)
            else:
                self.times[key] = time.time()
        return self.times[key]


if __name__ == "__main__":
    main(output_file="output.csv")
