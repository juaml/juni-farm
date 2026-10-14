"""Provide a datagrabber for CAT-processed confounds in the HCP dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

import warnings
from itertools import product
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AnyUrl, BeforeValidator, model_validator

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import (
    ConfoundsFormat,
    DataladHCP1200,
    DataType,
    MultipleDataGrabber,
    PatternDataladDataGrabber,
)
from junifer.typing import DataGrabberPatterns, Elements
from junifer.utils import ensure_list, raise_error


__all__ = [
    "JuselessDataladHCP1200",
    "JuselessDataladHCP1200CAT",
    "JuselessDataladHCP1200CATConfounds",
]


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


_tasks = Literal[
    "REST1",
    "REST2",
    "SOCIAL",
    "WM",
    "RELATIONAL",
    "EMOTION",
    "LANGUAGE",
    "GAMBLING",
    "MOTOR",
]

_all_tasks = [
    "REST1",
    "REST2",
    "SOCIAL",
    "WM",
    "RELATIONAL",
    "EMOTION",
    "LANGUAGE",
    "GAMBLING",
    "MOTOR",
]

_phase_encodings = Literal["LR", "RL"]


@register_datagrabber
class JuselessDataladHCP1200CATConfounds(PatternDataladDataGrabber):
    """Concrete implementation for CAT-processed HCP confounds.

    Parameters
    ----------
    datadir : pathlib.Path, optional
        That path where the datalad dataset will be cloned.
        If not specified, the datalad dataset will be cloned into a temporary
        directory.
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options, optional
        HCP task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"LR", "RL"} or list of the options, optional
        HCP phase encoding directions.
        By default, both are used.
    ica_fix : bool, optional
        Whether to retrieve the confounds of the data that was processed with
        ICA+FIX. Only "REST1" and "REST2" tasks are available with ICA+FIX
        (default False).

    """

    uri: AnyUrl = AnyUrl(
        "ria+file:///data/project/cat_preprocessed/dataladstore#~HCP-YA_conf"
    )
    types: Annotated[
        Literal[DataType.BOLD] | list[Literal[DataType.BOLD]],
        BeforeValidator(ensure_list),
    ] = [DataType.BOLD]  # noqa: RUF012
    tasks: Annotated[_tasks | list[_tasks], BeforeValidator(ensure_list)] = (
        _all_tasks
    )
    phase_encodings: Annotated[
        _phase_encodings | list[_phase_encodings],
        BeforeValidator(ensure_list),
    ] = ["LR", "RL"]  # noqa: RUF012
    patterns: DataGrabberPatterns = {  # noqa: RUF012
        "BOLD": {
            "confounds": {
                "pattern": (
                    "sub-{subject}/sub-{subject}_task-?fMRI{task}"
                    "{phase_encoding}{suffix}_desc-confounds_timeseries.tsv"
                ),
                "format": "adhoc",
                "mappings": {
                    "fmriprep": get_cat_to_fmriprep_mapping(),
                },
            },
        },
    }
    replacements: list[str] = ["subject", "task", "phase_encoding"]  # noqa: RUF012
    # Only grab the specified tasks and phase encodings
    _REPLACEMENT_FIELDS: ClassVar[dict[str, str]] = {
        "task": "tasks",
        "phase_encoding": "phase_encodings",
    }
    confounds_format: ConfoundsFormat = ConfoundsFormat.AdHoc
    # Only the confounds are grabbed, not the BOLD
    partial_pattern_ok: bool = True
    ica_fix: bool = False

    def validate_datagrabber_params(self) -> None:
        """Run extra logical validation for datagrabber."""
        if self.ica_fix:
            if not all(task in ["REST1", "REST2"] for task in self.tasks):
                raise_error(
                    "ICA+FIX is only available for 'REST1' and 'REST2' tasks."
                )
        suffix = "hp2000clean" if self.ica_fix else ""
        confounds = self.patterns["BOLD"]["confounds"]
        confounds["pattern"] = confounds["pattern"].replace("{suffix}", suffix)
        super().validate_datagrabber_params()

    def get_elements(self) -> Elements:
        """Implement fetching list of elements in the dataset.

        The task and the phase encoding are next to each other in the
        file names, so they cannot be found by searching the files: the
        elements are all the subjects with the tasks and phase encodings to
        grab, even if some of them have no data.

        Returns
        -------
        list
            The list of elements that can be grabbed in the dataset.

        """
        # there are some .git folders in the dataset that will be picked up
        # if we dont check whether "sub" is in name.
        subjects = sorted(
            x.name.split("-")[1]
            for x in self.fulldir.iterdir()
            if x.is_dir() and "sub" in x.name
        )
        values = {"subject": subjects, **self.get_replacement_values()}
        element_keys = self.get_element_keys()
        elements = list(product(*(values[k] for k in element_keys)))
        if len(element_keys) == 1:
            return [element[0] for element in elements]
        return elements


@register_datagrabber
class JuselessDataladHCP1200(DataladHCP1200):
    """Concrete implementation for datalad-based data fetching of HCP1200.

    This implementation uses a git repository on jusless, located at
    ``/data/group/appliedml``.

    Parameters
    ----------
    types : {"BOLD", "T1w", "Warp"} or list of the options, optional
        The data type(s) to grab.
    datadir : pathlib.Path, optional
        That path where the datalad dataset will be cloned.
        If not specified, the datalad dataset will be cloned into a temporary
        directory.
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options, optional
        HCP task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"LR", "RL"} or list of the options, optional
        HCP phase encoding directions.
        By default, both are used.
    ica_fix : bool, optional
        Whether to retrieve data that was processed with ICA+FIX.
        Only "REST1" and "REST2" tasks are available with ICA+FIX (default
        False).

    """

    uri: AnyUrl = AnyUrl(
        "file:///data/group/appliedml/datalad-datasets/"
        "human-connectome-project-openaccess.git"
    )


@register_datagrabber
class JuselessDataladHCP1200CAT(MultipleDataGrabber):
    """Concrete implementation for HCP1200 data with CAT-processed confounds.

    The datagrabbers are built from the parameters, so ``datagrabbers`` must
    not be given.

    Parameters
    ----------
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options, optional
        HCP task sessions.
        By default, all available task sessions are selected.
    phase_encodings : {"LR", "RL"} or list of the options, optional
        HCP phase encoding directions.
        By default, both are used.
    ica_fix : bool, optional
        Whether to retrieve data that was processed with ICA+FIX.
        Only "REST1" and "REST2" tasks are available with ICA+FIX (default
        False).

    """

    tasks: Annotated[_tasks | list[_tasks], BeforeValidator(ensure_list)] = (
        _all_tasks
    )
    phase_encodings: Annotated[
        _phase_encodings | list[_phase_encodings],
        BeforeValidator(ensure_list),
    ] = ["LR", "RL"]  # noqa: RUF012
    ica_fix: bool = False

    @model_validator(mode="before")
    @classmethod
    def build_datagrabbers(cls, data: Any) -> Any:
        """Build the HCP1200 and CAT confounds datagrabbers."""
        if not isinstance(data, dict):
            return data
        if "datagrabbers" in data:
            raise_error(
                "The datagrabbers of JuselessDataladHCP1200CAT are built "
                "from its parameters, they cannot be given."
            )
        params = {
            k: data[k] for k in ("tasks", "phase_encodings") if k in data
        }
        ica_fix = data.get("ica_fix", False)
        dg1 = JuselessDataladHCP1200(ica_fix=ica_fix, **params)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", module="junifer.datagrabber.pattern_validation_mixin"
            )
            dg2 = JuselessDataladHCP1200CATConfounds(ica_fix=ica_fix, **params)
        return {**data, "datagrabbers": [dg1, dg2]}


# test
if __name__ == "__main__":
    with JuselessDataladHCP1200CAT(tasks="REST1") as hcp_conf:
        all_elements = hcp_conf.get_elements()

        # this will run over all elements, so use keyboard interrupt
        # if convinced that it works
        for element in all_elements:
            print(element)
            out = hcp_conf[element]
            print(out)
