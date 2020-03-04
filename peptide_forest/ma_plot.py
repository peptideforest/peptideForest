import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

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
input_df = pd.read_csv("output.csv", index_col=0)

quant_df["MSMS_ID"] = quant_df["MSMS_ID"].str.lstrip("F0")
unique_spec_ids = input_df["Spectrum ID"].drop_duplicates()
ma_df = pd.DataFrame(index=unique_spec_ids)

q_val_cuts = sorted(
    [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(4, 1, -1)]
) + [1e-1]

final_df = pd.read_csv("output-final.csv")

final_df = peptide_forest.results.calc_all_final_q_vals(
    df=final_df, frac_tp=0.9, top_psm_only=False, initial_engine=None
)
for cut in q_val_cuts:
    all_eng = list(input_df["engine"].unique()) + ["RF-reg"]
    for eng in all_eng:
        eng_per_q_col = f"top_target_{eng}_at_{cut}"
        target_col = f"top_target_{eng}"
        marked_targets = peptide_forest.results.mark_top_targets(
            df=final_df, q_cut=cut
        )[[target_col, "Spectrum ID"]]
        marked_targets = marked_targets[marked_targets[target_col]]["Spectrum ID"]
        ma_df[eng_per_q_col] = False
        ma_df.loc[marked_targets, eng_per_q_col] = True

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

# Remove all nan rows
ma_df = ma_df[~ma_df[list(tmt_translation.values())].isna().all(axis=1)]

quotients = list(itertools.combinations(list(tmt_translation.values()), 2))

for ratio in quotients:
    m_col_name = f"M_{ratio[0]}_{ratio[1]}"
    a_col_name = f"A_{ratio[0]}_{ratio[1]}"
    ma_df[m_col_name] = np.log2(ma_df[ratio[0]]/ma_df[ratio[1]])
    ma_df.loc[~np.isfinite(ma_df[m_col_name]), m_col_name] = 0
    ma_df[a_col_name] = 0.5 * np.log2(ma_df[ratio[0]]*ma_df[ratio[1]])
    ma_df.loc[~np.isfinite(ma_df[a_col_name]), m_col_name] = 0

    for col in top_target_cols:
        sub_df = ma_df[ma_df[col]]
        plt.scatter(sub_df[a_col_name], sub_df[m_col_name])
        plot_name = f"MA_plot_{col}_{ratio[0]}_{ratio[1]}"
        plt.title(plot_name)
        path = f"plots/ma/{plot_name}.png"
        plt.savefig(path, format="png")
        plt.close()
