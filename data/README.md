# Data

The data are not in the repository. Download them and place them as below, or point `SORSOLA_ROOT` and `DRIAMS_ROOT` elsewhere.

| Cohort | Source | *E. coli* isolates analyzed |
|---|---|---|
| Sant'Orsola | Zenodo, [10.5281/zenodo.20087613](https://doi.org/10.5281/zenodo.20087613) | 187 of 191 (four lack a cefotaxime result) |
| DRIAMS-A, B, C, D | Dryad, [10.5061/dryad.bzkh1899q](https://doi.org/10.5061/dryad.bzkh1899q) | 4,761; 213; 882; 1,936 |

```
data/
  s_orsola/                        # the Zenodo record, extracted
    metadata.txt
    spectra/<ID>.txt
    data_bin_3.csv                 # written by `make data` (MaldiAMRKit 0.7.0)
  DRIAMS-A/                        # 2015 to 2018
    binned_6000/<year>/<code>.txt
    id/<year>/<year>_clean.csv
    id/<year>/<year>_strat.csv     # patient and case identifiers
  DRIAMS-B/  DRIAMS-C/  DRIAMS-D/  # 2018, same layout without _strat.csv
```

From DRIAMS only the `binned_6000` spectra and the `id` tables are needed. EUCAST "I" results count as susceptible at every site.
