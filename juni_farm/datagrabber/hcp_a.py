"""Provide a datagrabber for the HCP Aging dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from itertools import product
from typing import Annotated, ClassVar, Literal

from pydantic import AnyUrl, BeforeValidator

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import (
    DataladDataGrabber,
    DataType,
    PatternDataGrabber,
)
from junifer.typing import DataGrabberPatterns, Elements
from junifer.utils import ensure_list


__all__ = ["HCPA", "DataladHCPA"]


_types = Literal[DataType.BOLD, DataType.T1w, DataType.Warp]

_tasks = Literal["REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"]

_phase_encodings = Literal["AP", "PA"]


@register_datagrabber
class HCPA(PatternDataGrabber):
    """Concrete implementation for HCP Aging dataset.

    Parameters
    ----------
    types : {"BOLD", "T1w", "Warp"} or list of the options, optional
        The data type(s) to grab.
    datadir : pathlib.Path
        The path where the data is stored.
    tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"}
        or list of the options, optional
        HCP aging task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP aging phase encoding directions.
        By default, both are used.

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
        DataType.T1w,
        DataType.Warp,
    ]
    tasks: Annotated[_tasks | list[_tasks], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        "REST1",
        "REST2",
        "CARIT",
        "FACENAME",
        "VISMOTOR",
    ]
    phase_encodings: Annotated[
        _phase_encodings | list[_phase_encodings],
        BeforeValidator(ensure_list),
    ] = ["AP", "PA"]  # noqa: RUF012
    patterns: DataGrabberPatterns = {  # noqa: RUF012
        "BOLD": {
            "pattern": (
                "{subject}_V1_MR/MNINonLinear/"
                "Results/?fMRI_{task}_{phase_encoding}/"
                "?fMRI_{task}_{phase_encoding}_hp0_clean.nii.gz"
            ),
            "space": "MNI152NLin6Asym",
        },
        "T1w": {
            "pattern": "{subject}_V1_MR/T1w/T1w_acpc_dc_restore.nii.gz",
            "space": "native",
        },
        "Warp": [
            {
                "pattern": (
                    "{subject}_V1_MR/MNINonLinear/xfms/"
                    "standard2acpc_dc.nii.gz"
                ),
                "src": "MNI152NLin6Asym",
                "dst": "native",
                "warper": "fsl",
            },
            {
                "pattern": (
                    "{subject}_V1_MR/MNINonLinear/xfms/"
                    "acpc_dc2standard.nii.gz"
                ),
                "src": "native",
                "dst": "MNI152NLin6Asym",
                "warper": "fsl",
            },
        ],
    }
    replacements: list[str] = ["subject", "task", "phase_encoding"]  # noqa: RUF012
    # Only grab the specified tasks and phase encodings
    _REPLACEMENT_FIELDS: ClassVar[dict[str, str]] = {
        "task": "tasks",
        "phase_encoding": "phase_encodings",
    }

    def _replace_patterns_glob(self, element: dict, pattern: str) -> str:
        """Replace ``pattern`` with the ``element`` so it can be globbed.

        The BOLD data of the resting-state tasks is in ``rfMRI_*`` and of the
        other tasks in ``tfMRI_*``, so the path of an element is built
        without searching the files, which are in sub-datasets in
        :class:`.DataladHCPA`.

        Parameters
        ----------
        element : dict
            The element to be used in the replacement.
        pattern : str
            The pattern to be replaced.

        Returns
        -------
        str
            The pattern with the element replaced.

        """
        if "task" in element:
            prefix = "rfMRI" if "REST" in element["task"] else "tfMRI"
            pattern = pattern.replace("?fMRI_", f"{prefix}_")
        return super()._replace_patterns_glob(element, pattern)


@register_datagrabber
class DataladHCPA(DataladDataGrabber, HCPA):
    """Concrete implementation for datalad-based HCP Aging dataset.

    Parameters
    ----------
    types : {"BOLD", "T1w", "Warp"} or list of the options, optional
        The data type(s) to grab.
    datadir : pathlib.Path, optional
        That path where the datalad dataset will be cloned.
        If not specified, the datalad dataset will be cloned into a temporary
        directory.
    tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"}
        or list of the options, optional
        HCP aging task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP aging phase encoding directions.
        By default, both are used.

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
        DataType.T1w,
        DataType.Warp,
    ]
    uri: AnyUrl = AnyUrl(
        "ria+http://hcp-a.ds.inm7.de#431e4eb7-72c3-423d-b402-84ede40c03a2"
    )

    # Needed here as HCP Aging's subjects are sub-datasets, so will not be
    # found when elements are checked.
    @property
    def skip_file_check(self) -> bool:
        """Skip file check existence."""
        return True

    def get_elements(self) -> Elements:
        """Implement fetching list of elements in the dataset.

        The subjects are sub-datasets, whose files are only there once they
        are installed, so the files are not searched: the elements are all
        the subjects (the directories of the dataset) with the tasks and
        phase encodings to grab, even if some of them have no data.

        Returns
        -------
        list
            The list of elements that can be grabbed in the dataset.

        """
        subjects = sorted(
            x.name.split("_")[0]
            for x in self.fulldir.iterdir()
            if "V1_MR" in x.name
        )
        values = {"subject": subjects, **self.get_replacement_values()}
        element_keys = self.get_element_keys()
        elements = list(product(*(values[k] for k in element_keys)))
        if len(element_keys) == 1:
            return [element[0] for element in elements]
        return elements


if __name__ == "__main__":
    with DataladHCPA() as dg:
        elems = dg.get_elements()
        test_data = dg[elems[0]]
        print(test_data)
