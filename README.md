# juni-farm
A repository of curated python files to extend and customise junifer functionality or pipeline components.

Important links:
- [Junifer on GitHub](https://github.com/juaml/junifer)
- [Junifer's documentation](https://juaml.github.io/junifer/main/index.html)
- [Datalad python documentation](https://docs.datalad.org/en/stable/modref.html)
- [Datalad handbook](https://handbook.datalad.org/en/latest/)

# Additional Datagrabbers:

Find the additional datagrabbers that `juni-farm` provides in the `juni-farm/datagrabber/` directory.
The following is a list of datagrabbers already implemented here. If you implement another, add it
to this list:

- [A datalad datagrabber for Felix' HCP confounds dataset](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_ya_confounds_cat.py#L58) in `juni-farm/datagrabber/hcp_ya_confounds_cat.py`
- [A datalad datagrabber that combines the junifer DataladHCP1200 datagrabber with the above confound dataset](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_ya_confounds_cat.py#L220) in `juni-farm/datagrabber/hcp_ya_confounds_cat.py`
- [A datalad datagrabber for the HCP aging dataset](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_aging.py#L162) in `juni-farm/datagrabber/hcp_aging.py`
- [A datalad datagrabber for the HCP Early Psychosis dataset](https://github.com/juaml/juni-farm/blob/main/juni_farm/datagrabber/hcp_early_psychosis.py#L19) in `juni-farm/datagrabber/hcp_early_psychosis.py`
