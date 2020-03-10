import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic

import peptide_forest

tmt_translation = {
    "62": "126",
    "63": "127L",
    "64": "127H",
    "65": "128L",
    "66": "128H",
    "67": "129L",
    "68": "129H",
    "69": "130L",
    "70": "130H",
    "71": "131L",
}

quant_df = pd.read_csv(
    "data/_pep_quant_data/04854_F1_R8_P0109699E13_pep_quant_data.txt",
    sep="\t",
    lineterminator="\n",
)

final_df = pd.read_csv("output.csv")

quant_df["MSMS_ID"] = quant_df["MSMS_ID"].str.lstrip("F0")
unique_spec_ids = final_df["Spectrum ID"].drop_duplicates()
ma_df = pd.DataFrame(index=unique_spec_ids)

# q_val_cuts = sorted(
#     [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(4, 1, -1)]
# ) + [1e-1]
q_val_cuts = sorted([float(f"1e-{j}") for j in np.arange(4, 1, -1)]) + [1e-1]


final_df = peptide_forest.results.calc_all_final_q_vals(
    df=final_df, frac_tp=0.9, top_psm_only=False, initial_engine=None
)

all_eng = [c.split("Score_processed_")[1] for c in final_df.columns if "Score_processed" in c]
for cut in q_val_cuts:
    for eng in all_eng:
        eng_per_q_col = f"top_target_{eng}_at_{cut}"
        target_col = f"top_target_{eng}"
        marked_targets = peptide_forest.results.mark_top_targets(
            df=final_df, q_cut=cut
        )[[target_col, "Spectrum ID"]]
        marked_targets = marked_targets[marked_targets[target_col]]["Spectrum ID"]
        ma_df[eng_per_q_col] = False
        ma_df.loc[marked_targets, eng_per_q_col] = True

# Add species column
inds = ma_df.index.to_list()
spec_species = final_df.loc[final_df["Spectrum ID"].drop_duplicates().isin(inds).index, ["Protein ID", "Spectrum ID"]]
ma_df.loc[spec_species["Spectrum ID"].to_list(), "species"] = spec_species["Protein ID"].to_list()
ma_df.loc[ma_df["species"].str.contains("OS=Escherichia coli") & ~ma_df["species"].str.contains("OS=Homo sapiens"), "species"] = "E_coli"
ma_df.loc[~ma_df["species"].str.contains("OS=Escherichia coli") & ma_df["species"].str.contains("OS=Homo sapiens"), "species"] = "H_sapiens"
ma_df.loc[~ma_df["species"].str.contains("E_coli|H_sapiens")] = "Other"

# Drop rows that never appear as top target
top_target_cols = [c for c in ma_df.columns if "top_target" in c]
ma_df[top_target_cols] = ma_df[top_target_cols].astype(bool)
ma_df = ma_df[ma_df[top_target_cols].any(axis=1)]

for label, tmt in tmt_translation.items():
    values = quant_df[quant_df["ISOTOPELABEL_ID"] == int(label)][
        ["MSMS_ID", "QUANTVALUE"]
    ].astype({"MSMS_ID": "int64", "QUANTVALUE": "float64"})
    # Remove missing spectra
    values = values[values["MSMS_ID"].isin(ma_df.index)]
    ma_df.loc[values["MSMS_ID"], tmt] = values["QUANTVALUE"].to_list()

# Remove all SpecIDs where quant value is 0 in a mixed column
mixed_cols = ["127L", "128L", "129L", "130L", "131L"]
ma_df = ma_df[~ma_df[mixed_cols].any(axis=1) == 0]

# Remove all nan rows
ma_df = ma_df[~ma_df[list(tmt_translation.values())].isna().all(axis=1)]

#quotients = list(itertools.combinations(list(tmt_translation.values()), 2))
quotients = ("127L", "130L")
for species in ["H_sapiens", "E_coli"]:
    species_df = ma_df[ma_df["species"]==species]
    for cut in q_val_cuts:
        fig, ax = plt.subplots()
        cols_oi = [f"top_target_{eng}_at_{cut}" for eng in all_eng]
        for c in cols_oi:
            sub_df = species_df[species_df[c]]
            df = pd.DataFrame()
            df["x_axis"] = 1e7/(sub_df[quotients[0]] + sub_df[quotients[1]])
            df["y_axis"] = np.log2(sub_df[quotients[0]]/sub_df[quotients[1]])
            df.loc[~np.isfinite(df["y_axis"]), "y_axis"] = 0
            df.loc[~np.isfinite(df["x_axis"]), "x_axis"] = 0

            s, e, _ = binned_statistic(df["x_axis"], df["y_axis"], statistic='mean', bins=np.logspace(-2, 1, 15))

            bins = pd.DataFrame(data=[e[:-1] + np.diff(e) / 2, s]).transpose()
            bins.columns = ["x_axis", "y_axis"]
            bins = bins[bins["x_axis"] < 1.5]
            bins.dropna(inplace=True)
            sns.regplot(x="x_axis", y="y_axis", data=bins, logistic=True, ax=ax, ci=None, label=c.split("top_target_")[1].split("_at")[0])

        ax.set_xlim(0.01, 2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        plt.xscale("log")
        plt.title(f"{quotients}_{species}_at_{cut}")
        plt.savefig(f"plots/ma/{quotients}_{species}_at_{cut}.png")
        #plt.show()
