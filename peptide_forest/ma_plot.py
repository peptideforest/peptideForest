import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

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

expected_values = {
    "126": (0, 1),
    "127L": (1, 1),
    "127H": (1, 0),
    "128L": (0.5, 1),
    "128H": (0.5, 0),
    "129L": (0.2, 1),
    "129H": (0.2, 0),
    "130L": (0.1, 1),
    "130H": (0.1, 0),
    "131L": (1, 1),
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

all_eng = [
    c.split("Score_processed_")[1] for c in final_df.columns if "Score_processed" in c
]
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
spec_species = final_df.loc[
    final_df["Spectrum ID"].drop_duplicates().isin(inds).index,
    ["Protein ID", "Spectrum ID"],
]
ma_df.loc[spec_species["Spectrum ID"].to_list(), "species"] = spec_species[
    "Protein ID"
].to_list()
ma_df.loc[
    ma_df["species"].str.contains("OS=Escherichia coli")
    & ~ma_df["species"].str.contains("OS=Homo sapiens"),
    "species",
] = "E_coli"
ma_df.loc[
    ~ma_df["species"].str.contains("OS=Escherichia coli")
    & ma_df["species"].str.contains("OS=Homo sapiens"),
    "species",
] = "H_sapiens"
ma_df.loc[~ma_df["species"].str.contains("E_coli|H_sapiens")] = "Other"

# Drop rows that never appear as top target
top_target_cols = [c for c in ma_df.columns if "top_target" in c]
ma_df[top_target_cols] = ma_df[top_target_cols].astype(bool)
ma_df = ma_df[ma_df[top_target_cols].any(axis=1)]

# Generate all and any engines
for cut in q_val_cuts:
    eng_cols_per_cut = [f"top_target_{engine}_at_{cut}" for engine in all_eng]

    ma_df[f"top_target_any_engine_at_{cut}"] = False
    ma_df.loc[
        ma_df[eng_cols_per_cut].any(axis=1), f"top_target_any_engine_at_{cut}"
    ] = True

    ma_df[f"top_target_all_engines_at_{cut}"] = False
    ma_df.loc[
        ma_df[eng_cols_per_cut].all(axis=1), f"top_target_all_engines_at_{cut}"
    ] = True

all_eng = all_eng + ["all_engines", "any_engine"]

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

quotients = list(itertools.combinations(list(tmt_translation.values()), 2))

# [TRISTAN] temp
all_eng = ["all_engines", "any_engine", "RF-reg", "omssa_2_1_9"]

for ratio in quotients:
    for species in ["H_sapiens", "E_coli"]:
        if (
            species == "E_coli"
            and expected_values[ratio[1]][0] != 0
            and expected_values[ratio[0]][0] != 0
        ):
            exp_y = np.log2(expected_values[ratio[0]][0] / expected_values[ratio[1]][0])
        elif (
            species == "H_sapiens"
            and expected_values[ratio[1]][1] != 0
            and expected_values[ratio[0]][1] != 0
        ):
            exp_y = np.log2(expected_values[ratio[0]][1] / expected_values[ratio[1]][1])
        else:
            continue
        species_df = ma_df[ma_df["species"] == species]
        for cut in q_val_cuts:
            plot = plt.figure(figsize=(10, 10))
            cols_oi = [f"top_target_{eng}_at_{cut}" for eng in all_eng]
            df_plot = pd.DataFrame()
            for c in cols_oi:
                if species_df[c].sum() <= 1:
                    continue
                sub_df = species_df[species_df[c]]
                df = pd.DataFrame()
                df["x_axis"] = 1e7 / (sub_df[ratio[0]] + sub_df[ratio[1]])
                df["y_axis"] = np.log2(sub_df[ratio[0]] / sub_df[ratio[1]])
                df["engine"] = c.split("top_target_")[1].split("_at")[0]
                df = df[df["x_axis"] < 2]
                df.loc[~np.isfinite(df["y_axis"]), "y_axis"] = 0
                df.loc[~np.isfinite(df["x_axis"]), "x_axis"] = 0
                if len(df_plot) == 0:
                    df_plot = df
                else:
                    df_plot = pd.concat([df, df_plot])

            plot = sns.lmplot(
                x="x_axis",
                y="y_axis",
                hue="engine",
                data=df_plot,
                x_bins=np.logspace(-2, 1, 30),
                scatter_kws={"s": 5, "alpha": 0.5},
                x_ci="ci",
                markers="x",
                legend_out=True
            )
            plt.tight_layout()
            plot.ax.axhline(
                exp_y, color="black", linestyle="--", linewidth=2, alpha=0.5
            )
            plot.ax.set_xlim(0.01, 2)
            plot.set_axis_labels(
                f"1e7 / sum {ratio[0]}+{ratio[1]}", f"log2 {ratio[0]}x{ratio[1]}"
            )
            plot.set(xscale="log")
            plt.title(f"{ratio[0]}_{ratio[1]}_{species}_at_{cut}")
            # plot.ax.text(
            #     0,
            #     exp_y,
            #     f"Expected {species}",
            #     fontsize=7,
            #     va="center",
            #     ha="right",
            #     backgroundcolor="w",
            # )
            plt.savefig(f"plots/ma/{ratio}_{species}_at_{cut}.png")
            plt.show()
