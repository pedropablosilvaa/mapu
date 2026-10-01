"""
Built-in datasets for mapu, mirroring the datasets available in R's vegan package.
"""

import numpy as np
import pandas as pd


def load_dune() -> pd.DataFrame:
    """
    Load the Dune Meadow Vegetation dataset.

    This is the classic dataset from R's vegan package:
    ``data(dune)`` — Vegetation and Environment in Dutch Dune Meadows.
    It contains observations of 30 plant species at 20 sites.

    Returns
    -------
    pd.DataFrame
        A 20×30 DataFrame (sites × species) with integer abundance counts.

    References
    ----------
    Jongman, R.H.G, ter Braak, C.J.F & van Tongeren, O.F.R. (1987).
    Data Analysis in Community and Landscape Ecology.
    Pudoc, Wageningen.
    """
    species = [
        "Achimill",
        "Agrostol",
        "Airaprae",
        "Alopgeni",
        "Anthodor",
        "Bellpere",
        "Bromhord",
        "Chenalbu",
        "Cirsarve",
        "Comapalu",
        "Eleopalu",
        "Elymrepe",
        "Empenigr",
        "Hyporadi",
        "Juncarti",
        "Juncbufo",
        "Lolipere",
        "Planlanc",
        "Poaprat",
        "Poatriv",
        "Ranuflam",
        "Rumeacet",
        "Sagiproc",
        "Salirepe",
        "Scorautu",
        "Trifprat",
        "Trifrepe",
        "Vicilath",
        "Bracruta",
        "Callcusp",
    ]

    # Data verified against R vegan::dune (R 4.x, vegan 2.6+)
    # fmt: off
    data = np.array([
        # Site 1
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 7, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Site 2
        [3, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 4, 7, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0],
        # Site 3
        [0, 4, 0, 7, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 5, 6, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0],
        # Site 4
        [0, 8, 0, 2, 0, 2, 3, 0, 2, 0, 0, 4, 0, 0, 0, 0, 5, 0, 4, 5, 0, 0, 5, 0, 2, 0, 1, 0, 2, 0],
        # Site 5
        [2, 0, 0, 0, 4, 2, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 5, 2, 6, 0, 5, 0, 0, 3, 2, 2, 0, 2, 0],
        # Site 6
        [2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 3, 4, 0, 6, 0, 0, 3, 5, 5, 0, 6, 0],
        # Site 7
        [2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 5, 4, 5, 0, 3, 0, 0, 3, 2, 2, 0, 2, 0],
        # Site 8
        [0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4, 4, 2, 0, 2, 0, 3, 0, 2, 0, 2, 0],
        # Site 9
        [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 4, 4, 2, 0, 4, 5, 0, 2, 2, 0, 2, 0, 3, 0, 2, 0],
        # Site 10
        [4, 0, 0, 0, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 4, 4, 0, 0, 0, 0, 3, 0, 6, 1, 2, 0],
        # Site 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 7, 3, 4, 0, 0, 0, 2, 0, 5, 0, 3, 2, 4, 0],
        # Site 12
        [0, 4, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 2, 4, 0, 2, 0, 3, 0, 4, 0],
        # Site 13
        [0, 5, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 9, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0],
        # Site 14
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 6, 0, 0, 4],
        # Site 15
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 4, 0],
        # Site 16
        [0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 4, 3],
        # Site 17
        [2, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        # Site 18
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 0, 0, 0, 0, 3, 5, 0, 2, 1, 6, 0],
        # Site 19
        [0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 6, 0, 2, 0, 3, 0],
        # Site 20
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 5, 2, 0, 0, 0, 4, 3],
    ], dtype=float)
    # fmt: on

    index = list(range(1, 21))  # Sites 1–20
    return pd.DataFrame(data, index=index, columns=species)


def load_dune_env() -> pd.DataFrame:
    """
    Load the environmental variables for the Dune Meadow dataset.

    This is the companion dataset to ``load_dune()`` from R's vegan package:
    ``data(dune.env)`` — Environmental data for the 20 dune meadow sites.

    Variables
    ---------
    A1 : float
        Thickness of the soil A1 horizon (cm).
    Moisture : str
        Ordered moisture class: "1" < "2" < "4" < "5".
    Management : str
        Management type, one of: "BF" (Biological Farming),
        "HF" (Hobby Farming), "NM" (Nature Conservation Management),
        "SF" (Standard Farming).
    Use : str
        Ordered land use: "Hayfield" < "Haypastu" < "Pasture".
    Manure : str
        Ordered manure class: "0" < "1" < "2" < "3" < "4".

    Returns
    -------
    pd.DataFrame
        A 20×5 DataFrame (sites × environmental variables).

    References
    ----------
    Jongman, R.H.G, ter Braak, C.J.F & van Tongeren, O.F.R. (1987).
    Data Analysis in Community and Landscape Ecology.
    Pudoc, Wageningen.
    """
    data = {
        "A1": [
            2.8,
            3.5,
            4.3,
            4.2,
            6.3,
            4.3,
            2.8,
            4.2,
            3.7,
            3.3,
            3.5,
            5.8,
            6.0,
            9.3,
            11.5,
            5.7,
            4.0,
            4.6,
            3.7,
            3.5,
        ],
        "Moisture": [
            "1",
            "1",
            "2",
            "2",
            "1",
            "1",
            "1",
            "5",
            "4",
            "2",
            "1",
            "4",
            "5",
            "5",
            "5",
            "5",
            "2",
            "1",
            "5",
            "5",
        ],
        "Management": [
            "SF",
            "BF",
            "SF",
            "SF",
            "HF",
            "HF",
            "HF",
            "HF",
            "HF",
            "BF",
            "BF",
            "SF",
            "SF",
            "NM",
            "NM",
            "SF",
            "NM",
            "NM",
            "NM",
            "NM",
        ],
        "Use": [
            "Haypastu",
            "Haypastu",
            "Haypastu",
            "Haypastu",
            "Hayfield",
            "Haypastu",
            "Pasture",
            "Pasture",
            "Hayfield",
            "Hayfield",
            "Pasture",
            "Haypastu",
            "Haypastu",
            "Pasture",
            "Haypastu",
            "Pasture",
            "Hayfield",
            "Hayfield",
            "Hayfield",
            "Hayfield",
        ],
        "Manure": [
            "4",
            "2",
            "4",
            "4",
            "2",
            "2",
            "3",
            "3",
            "1",
            "1",
            "1",
            "2",
            "3",
            "0",
            "0",
            "3",
            "0",
            "0",
            "0",
            "0",
        ],
    }
    index = list(range(1, 21))
    return pd.DataFrame(data, index=index)
