import csv
import glob
import os
import sys
from collections import defaultdict as ddict


def main(quant_folder, ucsv_folder):
    """
    Add mScore values from quantvalue .csvs

    Usage

    python add_mScore.py <folder to quant csvs> <folder to unified_csvs>
    """
    qlookup = ddict(dict)
    files = glob.glob(os.path.join(quant_folder, "quant*.csv"))
    for f in sorted(files):
        print(f"Packing {f} into lookup structure", end="\r")
        with open(f) as qcsv:
            cdReader = csv.DictReader(qcsv)
            for d in cdReader:
                m_key = "{molecule}|{charge}".format(**d)
                qlookup[m_key][int(d["scan_id"])] = float(d["mScore"])

    ident_files = glob.glob(os.path.join(ucsv_folder, "*.csv"))
    for ifile in ident_files:
        with open(f'{ifile.split(".csv")[0]}_mscore.csv', "w") as ocvs:
            with open(ifile) as icvs:
                cdReader = csv.DictReader(icvs)
                cdWriter = csv.DictWriter(
                    ocvs, fieldnames=cdReader.fieldnames + ["mScore"]
                )
                cdWriter.writeheader()
                for n, d in enumerate(cdReader):
                    print(f"{n}", end="\r")
                    if d["Modifications"] == "":
                        m_key = "{Sequence}|{Charge}".format(**d)
                    else:
                        m_key = "{Sequence}#{Modifications}|{Charge}".format(**d)
                    d["mScore"] = 0
                    ident_spec_id = int(d["Spectrum ID"])
                    for _ in range(-20, 21):
                        scan = ident_spec_id + _
                        mscore = qlookup[m_key].get(scan, 0)
                        if mscore > d["mScore"]:
                            d["mScore"] = mscore
                    cdWriter.writerow(d)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(main.__doc__)
    else:
        main(sys.argv[1], sys.argv[2])
