# mc_lilliefors_exp

Python code for estimating the magnitude of completeness Mc of an earthquake catalog using the Lilliefors test combined with a truncated-exponential dithering technique.

**Files**
1. `mc_lilliefors_exp.py` --> This module contains the core functions for Mc estimation.
2. `Test_Lillie.ipynb` --> A Jupyter Notebook demonstrating the usage of the `mc_lilliefors_exp.py` module.

**Requirements** 

The script requires the following Python libraries: 
- numpy (1.26.3 or later) 
- statsmodels (0.14.1 or later)

**Reference article**

```bash
@article{stallone2026exponentiality,
  author  = {Angela Stallone and Ilaria Spassiani},
  title   = {Correcting exponentiality test for binned earthquake magnitudes},
  journal = {Seismica},
  volume  = {5},
  number  = {1},
  year    = {2026},
  month   = feb,
  doi     = {10.26443/seismica.v5i1.2257},
  url     = {https://doi.org/10.26443/seismica.v5i1.2257}
}
```

[doi.org/10.26443/seismica.v5i1.2257](https://doi.org/10.26443/seismica.v5i1.2257)
