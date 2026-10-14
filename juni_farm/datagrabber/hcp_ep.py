"""Provide a datagrabber for the HCP Early Psychosis dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from itertools import product
from typing import Annotated, ClassVar, Literal

from pydantic import AnyUrl, BeforeValidator

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import (
    ConfoundsFormat,
    DataladDataGrabber,
    DataType,
    PatternDataGrabber,
)
from junifer.typing import DataGrabberPatterns, Elements
from junifer.utils import ensure_list


__all__ = ["HCPEP", "JuselessDataladHCPEP"]


def get_cat_to_fmriprep_mapping():
    """Map variables in CAT output to fmriprep variables.

    Returns
    -------
    dict
        keys (CAT variables) and values (corresponding fMRIprep variables).

    """
    # overarching variables
    terms_cat = ["WM", "CSF", "GS"]
    terms_fmriprep = ["white_matter", "csf", "global_signal"]

    mapping = {}

    for cat, fmriprep in zip(terms_cat, terms_fmriprep):
        mapping[cat] = fmriprep
        mapping[f"{cat}^2"] = f"{fmriprep}_power2"

    # take care of motion parameters
    # TODO: Felix' dataset uses rigid body parameters 1 to 6 but i am not sure
    # which number (1-6) correspnds to translations and rotations (and x, y, z)
    # respectively; for regular confound removal this should not matter
    # because all confounds will be selected and used in the regression
    # but will be good to have this implemented correctly anyways
    motion_terms_fmriprep = ["rot", "trans"]
    motion_directions = ["x", "y", "z"]
    for i_iter, (term, direction) in enumerate(
        product(motion_terms_fmriprep, motion_directions)
    ):
        mapping[f"RP.{i_iter+1}"] = f"{term}_{direction}"
        mapping[f"RP^2.{i_iter+1}"] = f"{term}_{direction}_power2"
        mapping[f"DRP.{i_iter+1}"] = f"{term}_{direction}_derivative1"
        mapping[f"DRP^2.{i_iter+1}"] = f"{term}_{direction}_derivative1_power2"

    return mapping


_types = Literal[DataType.BOLD, DataType.T1w]

_tasks = Literal["REST1", "REST2"]

_phase_encodings = Literal["AP", "PA"]


@register_datagrabber
class HCPEP(PatternDataGrabber):
    """Concrete implementation for HCP Early Psychosis dataset.

    Parameters
    ----------
    types : {"BOLD", "T1w"} or list of the options, optional
        The data type(s) to grab (default "BOLD"). There is no T1w in native
        space, only in MNI152NLin6Asym, and not for all the subjects.
    datadir : pathlib.Path
        The path where the data is stored.
    tasks : {"REST1", "REST2"} or list of the options, optional
        HCP early psychosis task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP early psychosis phase encoding directions.
        By default, both are used.

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
    ]
    tasks: Annotated[_tasks | list[_tasks], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        "REST1",
        "REST2",
    ]
    phase_encodings: Annotated[
        _phase_encodings | list[_phase_encodings],
        BeforeValidator(ensure_list),
    ] = ["AP", "PA"]  # noqa: RUF012
    patterns: DataGrabberPatterns = {  # noqa: RUF012
        "BOLD": {
            "pattern": (
                "{subject}_01_MR/MNINonLinear/"
                "Results/rfMRI_{task}_{phase_encoding}/"
                "rfMRI_{task}_{phase_encoding}_hp0_clean.nii.gz"
            ),
            "space": "MNI152NLin6Asym",
            "confounds": {
                "pattern": (
                    "HCP_EP_confounds/{subject}_01_MR/MNINonLinear/"
                    "Results/rfMRI_{task}_{phase_encoding}/"
                    "Confounds_{subject}_01_MR.tsv"
                ),
                "format": "adhoc",
                "mappings": {
                    "fmriprep": get_cat_to_fmriprep_mapping(),
                },
            },
        },
        "T1w": {
            "pattern": "{subject}_01_MR/MNINonLinear/T1w_restore.nii.gz",
            "space": "MNI152NLin6Asym",
        },
    }
    replacements: list[str] = ["subject", "task", "phase_encoding"]  # noqa: RUF012
    # Only grab the specified tasks and phase encodings
    _REPLACEMENT_FIELDS: ClassVar[dict[str, str]] = {
        "task": "tasks",
        "phase_encoding": "phase_encodings",
    }
    confounds_format: ConfoundsFormat = ConfoundsFormat.AdHoc

    def get_elements(self) -> Elements:
        """Implement fetching list of elements in the dataset.

        Returns
        -------
        list
            The list of elements in the dataset.

        """
        subjects = sorted(
            x.name.split("_")[0]
            for x in self.fulldir.iterdir()
            if "01_MR" in x.name
        )
        values = {"subject": subjects, **self.get_replacement_values()}
        element_keys = self.get_element_keys()
        elements = list(product(*(values[k] for k in element_keys)))
        if len(element_keys) == 1:
            return [element[0] for element in elements]
        return elements

    @property
    def skip_file_check(self) -> bool:
        """Skip file check existence."""
        return True


@register_datagrabber
class JuselessDataladHCPEP(DataladDataGrabber, HCPEP):
    """Concrete implementation for datalad-based HCP Early Psychosis dataset.

    Parameters
    ----------
    types : {"BOLD", "T1w"} or list of the options, optional
        The data type(s) to grab (default "BOLD"). There is no T1w in native
        space, only in MNI152NLin6Asym, and not for all the subjects.
    datadir : pathlib.Path, optional
        That path where the datalad dataset will be cloned.
        If not specified, the datalad dataset will be cloned into a temporary
        directory.
    tasks : {"REST1", "REST2"} or list of the options, optional
        HCP Early Psychosis task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP Early Psychosis phase encoding directions.
        By default, both are used.

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
    ]
    # The dataset is also on jugit
    # (git@jugit.fz-juelich.de:inm7/hcp_earlypsychosis_preprocessed), but
    # only the superdataset: the sub-datasets of the subjects are only on
    # juseless, so they cannot be installed from a clone of jugit.
    uri: AnyUrl = AnyUrl(
        "file:///data/project/hcp_earlypsychosis_preprocessing"
    )


if __name__ == "__main__":
    with JuselessDataladHCPEP() as dg:
        elems = dg.get_elements()
        print(elems)
        test_data = dg[elems[0]]
        print(test_data)
