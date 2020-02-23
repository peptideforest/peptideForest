import pandas as pd

import add_mScore

quant_data = pd.DataFrame(
    {
        "mScore": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "scan_id": [1, 2, 3, 4, 1, 19, 20, 21, 22, 23],
        "search param": [
            ">AAAA%moda<",
            ">AAAA%modb<",
            ">AAAA%modc<",
            ">DDDD%moda<",
            ">AAAA%moda;modb<",
            ">AAAA%moda<",
            ">AAAA%moda<",
            ">AAAA%moda<",
            ">AAAA%moda<",
            ">AAAA%moda<",
        ],
    }
)

df = pd.DataFrame(
    {
        "Spectrum ID": [1, 1, 1, 1, 1, 1, 2, 3, 1, 1],
        "search param": [
            ">AAAA%moda<",
            ">AAAA%modb<",
            ">AAAA%modc<",
            ">DDDD%moda<",
            ">AAAA%moda;modb<",
            ">AAAA%moda<",
            ">AAAA%moda<",
            ">AAAA%moda<",
            ">XXXX%moda<",
            ">AAAA%modX<",
        ],
    }
)


def test_get_mScore():
    df["mScore"] = df.apply(lambda x: add_mScore.get_mScore(x, qd=quant_data), axis=1)
    assert all(df["mScore"] == [8, 2, 3, 4, 5, 8, 9, 10, 0, 0])
