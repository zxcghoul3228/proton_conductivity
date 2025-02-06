import os
import json


def create_raw_pcond(root_cifs="./root_cifs"):
    """
    Create raw_pcond.json file with zeros as target values.

    Args:
        root_cifs: folder with .cif files
    """
    cif_idx = [cif[:-4] for cif in os.listdir(root_cifs) if cif.endswith(".cif")]
    trg_dict = dict(zip(cif_idx, [0] * len(cif_idx)))
    json.dump(trg_dict, open(os.path.join(root_cifs, "raw_pcond.json"), "w"))


def create_pcond_features(
    RH: float = 0.98,
    T: float = 353.0,
    Ka: float = -1.7,
    extra_proton: float = 0.0,
    root_dataset="./root_dataset",
):
    """
    Create test_pcond_features.json file with normalized global features (RH, 1000/T, Ka, extra_proton)

    Args:
        RH: relative humidity value in the range from 0 to 1
        T: temperature in Kelvin
        Ka: solvent dissociation constant (default = -1.7 (water))
        extra_proton: degree of protonation (default = 0.0 (not protonated))
        root_dataset: folder with dataset created prepare_data function
    """
    # Feature normalization
    _RH = (RH - 0.8625437873460142) / 0.16382520644832796
    _T = (1000 / T - 3.096828824472335) / 0.23282128469042473
    _Ka = (Ka - 0.05705342237061772) / 4.059215838170252
    _extra_proton = (extra_proton - 0.18447412353923207) / 0.387870366596446

    trg_dict = json.load(open(os.path.join(root_dataset, "test_pcond.json")))
    cif_idx = list(trg_dict.keys())
    fea_dict = dict(zip(cif_idx, [[_RH, _T, _Ka, _extra_proton]] * len(cif_idx)))
    json.dump(
        fea_dict, open(os.path.join(root_dataset, "test_pcond_features.json"), "w")
    )
