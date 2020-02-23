import pandas as pd
import glob
import os


def get_mScore(x, qd):
    mscore = max(
        qd[
            (qd["search param"].str.contains(x["search param"], regex=False))
            & (abs(qd["scan_id"] - x["Spectrum ID"]) <= 20)
        ]["mScore"],
        default=0,
    )
    return mscore


def add_mScores():
    path = "data/_notebooks/"

    files = glob.glob(os.path.join(path, "*.csv"))

    quant_data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    quant_data_split = quant_data["molecule"].str.split("#", expand=True)
    quant_data["sequence"] = quant_data_split[0]
    quant_data["modifications"] = quant_data_split[1]
    quant_data = quant_data.filter(["mScore", "sequence", "scan_id", "modifications"])

    quant_data["modifications"] = quant_data["modifications"].str.split(";")
    quant_data["modifications"] = quant_data["modifications"].apply(
        lambda x: sorted(x) if isinstance(x, list) else []
    )
    quant_data["modifications"] = quant_data["modifications"].str.join(";")
    quant_data["search param"] = (
        ">" + quant_data["sequence"] + "%" + quant_data["modifications"] + "<"
    )

    path_input = "data/"
    files_input = glob.glob(os.path.join(path_input, "*.csv"))

    for f in files_input:
        df = pd.read_csv(f)
        df["Modifications"] = df["Modifications"].str.split(";")
        df["Modifications"] = df["Modifications"].apply(
            lambda x: sorted(x) if isinstance(x, list) else []
        )
        df["Modifications"] = df["Modifications"].str.join(";")
        df["search param"] = ">" + df["Sequence"] + "%" + df["Modifications"] + "<"
        df["mScore"] = df.apply(lambda x: get_mScore(x, quant_data), axis=1)

        output_file = (
            f.split("/")[0] + "/_mScore/" + f.split("/")[1] + "_plus_mScores.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"Finished {f}.")


if __name__ == "__main__":
    add_mScores()
