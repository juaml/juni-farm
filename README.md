# juni-farm
A repository of curated python files to extend and customise junifer functionality or pipeline components.

Important links:
- [Junifer on GitHub](https://github.com/juaml/junifer)
- [Junifer's documentation](https://juaml.github.io/junifer/main/index.html)
- [Datalad python documentation](https://docs.datalad.org/en/stable/modref.html)
- [Datalad handbook](https://handbook.datalad.org/en/latest/)

# Additional Datagrabbers:

Find the additional datagrabbers that `juni-farm` provides in the `juni-farm/datagrabber/` directory.
Each file can be used on its own (e.g., with `with: <file>.py` in the YAML).
The datagrabbers are named `<Dataset>` for a local copy of the dataset and
`Datalad<Dataset>` for the datalad dataset, prefixed with `Juseless` if the
dataset is only available on juseless.

The following is a list of datagrabbers already implemented here. If you implement another, add it
to this list:

- [HCP1200 from the juseless clone](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp1200_cat.py) in `juni-farm/datagrabber/hcp1200_cat.py`: `JuselessDataladHCP1200`, the junifer `DataladHCP1200` datagrabber with the clone of the dataset on juseless
- [HCP1200 with Felix' CAT confounds](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp1200_cat.py) in `juni-farm/datagrabber/hcp1200_cat.py`:
  - `JuselessDataladHCP1200CATConfounds`: the CAT confounds dataset
  - `JuselessDataladHCP1200CAT`: combines `JuselessDataladHCP1200` with the CAT confounds
- [HCP1200 with CAT confounds, concatenating the task runs of each subject](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp1200_cat_concatenated.py) in `juni-farm/datagrabber/hcp1200_cat_concatenated.py`: `JuselessDataladHCP1200CATConcatenated`
- [HCP Aging](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_a.py) in `juni-farm/datagrabber/hcp_a.py`: `HCPA` and `DataladHCPA`
- [HCP Development](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_d.py) in `juni-farm/datagrabber/hcp_d.py`: `HCPD` and `DataladHCPD`
- [HCP Early Psychosis](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_ep.py) in `juni-farm/datagrabber/hcp_ep.py`: `HCPEP` and `JuselessDataladHCPEP`
