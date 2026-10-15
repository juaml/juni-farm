"""Provide datagrabbers for the fMRIPrep derivatives of HCP-EP."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from itertools import product
from pathlib import Path
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


__all__ = ["DataladHCPEPfMRIPrep", "HCPEPfMRIPrep"]


_types = Literal[
    DataType.BOLD,
    DataType.T1w,
    DataType.VBM_CSF,
    DataType.VBM_GM,
    DataType.VBM_WM,
    DataType.Warp,
]

_tasks = Literal["REST1", "REST2"]

_phase_encodings = Literal["AP", "PA"]

_anat_spaces = Literal["native", "MNI152NLin2009cAsym"]

# The BIDS run of each task
_TASK_RUNS = {"REST1": "01", "REST2": "02"}


@register_datagrabber
class HCPEPfMRIPrep(PatternDataGrabber):
    """Concrete implementation for the fMRIPrep derivatives of HCP-EP.

    The resting-state BOLD was preprocessed with fMRIPrep 25.1.4 and is only
    available in MNI152NLin2009cAsym (2 mm). The anatomical data is available
    in native space and in MNI152NLin2009cAsym (2 mm), with the warps
    between them, so the BOLD can be warped to native space.

    Parameters
    ----------
    types : {"BOLD", "T1w", "VBM_CSF", "VBM_GM", "VBM_WM", "Warp"} or \
            list of the options, optional
        The data type(s) to grab.
    datadir : pathlib.Path
        The path where the data is stored (the ``fmriprep`` directory).
    tasks : {"REST1", "REST2"} or list of the options, optional
        HCP Early Psychosis resting-state runs.
        By default, both are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP Early Psychosis phase encoding directions.
        By default, both are used.
    anat_space : {"native", "MNI152NLin2009cAsym"}, optional
        The space of the anatomical data (T1w, VBM_CSF, VBM_GM and VBM_WM).
        The BOLD is always in MNI152NLin2009cAsym
        (default "MNI152NLin2009cAsym").

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
        DataType.T1w,
        DataType.VBM_CSF,
        DataType.VBM_GM,
        DataType.VBM_WM,
        DataType.Warp,
    ]
    tasks: Annotated[_tasks | list[_tasks], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        "REST1",
        "REST2",
    ]
    phase_encodings: Annotated[
        _phase_encodings | list[_phase_encodings],
        BeforeValidator(ensure_list),
    ] = ["AP", "PA"]  # noqa: RUF012
    anat_space: _anat_spaces = "MNI152NLin2009cAsym"
    patterns: DataGrabberPatterns = {  # noqa: RUF012
        "BOLD": {
            "pattern": (
                "sub-{subject}/func/sub-{subject}_task-rest_"
                "dir-{phase_encoding}_run-{task}_"
                "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            ),
            "space": "MNI152NLin2009cAsym",
            "mask": {
                "pattern": (
                    "sub-{subject}/func/sub-{subject}_task-rest_"
                    "dir-{phase_encoding}_run-{task}_"
                    "space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
                ),
                "space": "MNI152NLin2009cAsym",
            },
            "confounds": {
                "pattern": (
                    "sub-{subject}/func/sub-{subject}_task-rest_"
                    "dir-{phase_encoding}_run-{task}_"
                    "desc-confounds_timeseries.tsv"
                ),
                "format": "fmriprep",
            },
            "reference": {
                "pattern": (
                    "sub-{subject}/func/sub-{subject}_task-rest_"
                    "dir-{phase_encoding}_run-{task}_"
                    "space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
                ),
            },
            "prewarp_space": "native",
        },
        "T1w": {
            "pattern": (
                "sub-{subject}/anat/sub-{subject}_{anat_space}"
                "desc-preproc_T1w.nii.gz"
            ),
            "mask": {
                "pattern": (
                    "sub-{subject}/anat/sub-{subject}_{anat_space}"
                    "desc-brain_mask.nii.gz"
                ),
            },
        },
        "VBM_CSF": {
            "pattern": (
                "sub-{subject}/anat/sub-{subject}_{anat_space}"
                "label-CSF_probseg.nii.gz"
            ),
        },
        "VBM_GM": {
            "pattern": (
                "sub-{subject}/anat/sub-{subject}_{anat_space}"
                "label-GM_probseg.nii.gz"
            ),
        },
        "VBM_WM": {
            "pattern": (
                "sub-{subject}/anat/sub-{subject}_{anat_space}"
                "label-WM_probseg.nii.gz"
            ),
        },
        "Warp": [
            {
                "pattern": (
                    "sub-{subject}/anat/sub-{subject}_"
                    "from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
                ),
                "src": "MNI152NLin2009cAsym",
                "dst": "native",
                "warper": "ants",
            },
            {
                "pattern": (
                    "sub-{subject}/anat/sub-{subject}_"
                    "from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
                ),
                "src": "native",
                "dst": "MNI152NLin2009cAsym",
                "warper": "ants",
            },
        ],
    }
    replacements: list[str] = ["subject", "task", "phase_encoding"]  # noqa: RUF012
    # Only grab the specified tasks and phase encodings
    _REPLACEMENT_FIELDS: ClassVar[dict[str, str]] = {
        "task": "tasks",
        "phase_encoding": "phase_encodings",
    }
    confounds_format: ConfoundsFormat = ConfoundsFormat.FMRIPrep

    def validate_datagrabber_params(self) -> None:
        """Run extra logical validation for datagrabber."""
        # Descriptor of the space in the anatomical file names
        anat_space = (
            ""
            if self.anat_space == "native"
            else "space-MNI152NLin2009cAsym_res-2_"
        )
        for t in ["T1w", "VBM_CSF", "VBM_GM", "VBM_WM"]:
            entries = [self.patterns[t]]
            if "mask" in self.patterns[t]:
                entries.append(self.patterns[t]["mask"])
            for entry in entries:
                entry["pattern"] = entry["pattern"].replace(
                    "{anat_space}", anat_space
                )
                entry["space"] = self.anat_space
        super().validate_datagrabber_params()

    def _replace_patterns_glob(self, element: dict, pattern: str) -> str:
        """Replace ``pattern`` with the ``element`` so it can be globbed.

        The tasks are the BIDS runs in the file names.

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
            element = {**element, "task": _TASK_RUNS[element["task"]]}
        return super()._replace_patterns_glob(element, pattern)

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
            x.name[4:]
            for x in self.fulldir.iterdir()
            if x.is_dir() and x.name.startswith("sub-")
        )
        values = {"subject": subjects, **self.get_replacement_values()}
        element_keys = self.get_element_keys()
        elements = list(product(*(values[k] for k in element_keys)))
        if len(element_keys) == 1:
            return [element[0] for element in elements]
        return elements


@register_datagrabber
class DataladHCPEPfMRIPrep(DataladDataGrabber, HCPEPfMRIPrep):
    """Concrete implementation for the datalad-based fMRIPrep HCP-EP.

    Parameters
    ----------
    types : {"BOLD", "T1w", "VBM_CSF", "VBM_GM", "VBM_WM", "Warp"} or \
            list of the options, optional
        The data type(s) to grab.
    datadir : pathlib.Path, optional
        That path where the datalad dataset will be cloned.
        If not specified, the datalad dataset will be cloned into a temporary
        directory.
    tasks : {"REST1", "REST2"} or list of the options, optional
        HCP Early Psychosis resting-state runs.
        By default, both are selected.
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP Early Psychosis phase encoding directions.
        By default, both are used.
    anat_space : {"native", "MNI152NLin2009cAsym"}, optional
        The space of the anatomical data (T1w, VBM_CSF, VBM_GM and VBM_WM).
        The BOLD is always in MNI152NLin2009cAsym
        (default "MNI152NLin2009cAsym").

    """

    types: Annotated[_types | list[_types], BeforeValidator(ensure_list)] = [  # noqa: RUF012
        DataType.BOLD,
        DataType.T1w,
        DataType.VBM_CSF,
        DataType.VBM_GM,
        DataType.VBM_WM,
        DataType.Warp,
    ]
    uri: AnyUrl = AnyUrl("ria+http://hcp-ep.ds.inm7.de#~fmriprep25.1.4")
    rootdir: Path = Path("fmriprep")

    # Needed here as the subjects are sub-datasets, so will not be found
    # when elements are checked.
    @property
    def skip_file_check(self) -> bool:
        """Skip file check existence."""
        return True
