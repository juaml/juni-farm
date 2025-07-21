"""Provide a datagrabber for CAT-processed confounds in the HCP dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from itertools import product
from pathlib import Path
from typing import Dict, List, Union
import warnings

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber
from junifer.datagrabber import (
    DataladDataGrabber,
    HCP1200,
    MultipleDataGrabber,
    PatternDataladDataGrabber,
)
from junifer.utils import raise_error


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


@register_datagrabber
class HCPCATConfounds(PatternDataladDataGrabber):
    """Concrete implementation for CAT-processed HCP confounds.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"} or list of the options, optional
        HCP task sessions. If None, all available task sessions are selected
        (default None).
    phase_encodings : {"LR", "RL"} or list of the options, optional
        HCP phase encoding directions. If None, both will be used
        (default None).
    **kwargs
        Keyword arguments passed to superclass.

    """

    def __init__(
        self,
        datadir: Union[str, Path] = None,
        tasks: Union[str, List[str], None] = None,
        phase_encodings: Union[str, List[str], None] = None,
        **kwargs,
    ) -> None:
        """Initialise the class."""
        # All tasks
        all_tasks = [
            "REST1",
            "REST2",
            "CARIT",
            "FACENAME",
            "VISMOTOR",
        ]
        # Set default tasks
        if tasks is None:
            self.tasks: List[str] = all_tasks
        # Convert single task into list
        else:
            if not isinstance(tasks, List):
                tasks = [tasks]

            # Check for invalid task(s)
            for task in tasks:
                if task not in all_tasks:
                    raise_error(
                        f"'{task}' is not a valid HCP-A fMRI task input. "
                        f"Valid task values can be any or all of {all_tasks}."
                    )
            self.tasks: List[str] = tasks

        # All phase encodings
        all_phase_encodings = ["AP", "PA"]
        # Set phase encodings
        if phase_encodings is None:
            phase_encodings = all_phase_encodings
        # Convert single phase encoding into list
        if isinstance(phase_encodings, str):
            phase_encodings = [phase_encodings]
        # Check for invalid phase encoding(s)
        for pe in phase_encodings:
            if pe not in all_phase_encodings:
                raise_error(
                    f"'{pe}' is not a valid HCP-A phase encoding. "
                    "Valid phase encoding can be any or all of "
                    f"{all_phase_encodings}."
                )

        # The types of data
        types = ["BOLD"]
        # The patterns
        patterns = {
            "BOLD": {
                "confounds": {
                    "pattern": (
                        "sub-{subject}/sub-{subject}_task-{task}"
                        "{phase_encoding}_desc-confounds_timeseries.tsv"
                    ),
                    "format": "adhoc",
                    "mappings": {
                        "fmriprep": get_cat_to_fmriprep_mapping(),
                    },
                },
            },
        }
        # The replacements
        replacements = [
            "subject",
            "task",
            "phase_encoding",
        ]
        uri = (
            "ria+file:///data/project/cat_preprocessed/"
            "dataladstore#~HCP-A_conf"
        )
        super().__init__(
            types=types,
            datadir=datadir,
            uri=uri,
            patterns=patterns,
            replacements=replacements,
            confounds_format="adhoc",
            **kwargs,
        )
        self.phase_encodings = phase_encodings

    def get_item(self, subject: str, task: str, phase_encoding: str) -> Dict:
        """Index one element in the dataset.

        Parameters
        ----------
        subject : str
            The subject ID.
        task : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
               "LANGUAGE", "GAMBLING", "MOTOR"}
            The task.
        phase_encoding : {"LR", "RL"}
            The phase encoding.

        Returns
        -------
        out : dict
            Dictionary of paths for each type of data required for the
            specified element.
        """
        # Resting task
        if "REST" in task:
            new_task = f"rfMRI{task}"
            new_phase_encoding = f"{phase_encoding}hp2000clean"
        else:
            new_task = f"tfMRI{task}"
            new_phase_encoding = phase_encoding

        return super().get_item(
            subject=subject, task=new_task, phase_encoding=new_phase_encoding
        )

    def get_elements(self) -> List:
        """Implement fetching list of elements in the dataset.

        Returns
        -------
        list
            The list of elements in the dataset.
        """
        # there are some .git folders in the dataset that will be picked up
        # if we dont check whether "sub" is in name.
        subjects = [
            x.name.split("-")[1]
            for x in self.datadir.iterdir()
            if x.is_dir() and "sub" in x.name
        ]
        elems = []
        for subject, task, phase_encoding in product(
            subjects, self.tasks, self.phase_encodings
        ):
            elems.append((subject, task, phase_encoding))

        return elems

@register_datagrabber
class HCPAging(PatternDataGrabber):
    """Concrete implementation for HCP Aging dataset.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"}
        or list of the options, optional
        HCP aging task sessions. If None, all available task sessions
        are selected (default None).
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP aging phase encoding directions. If None, both will be used
        (default None).
    **kwargs
        Keyword arguments passed to superclass.

    """

    def __init__(
        self,
        datadir: Union[str, Path] = None,
        tasks: Union[str, List[str], None] = None,
        phase_encodings: Union[str, List[str], None] = None,
        **kwargs,
    ) -> None:
        """Initialise the class."""
        # all tasks

        all_tasks = ["REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"]
        patterns = {
            "BOLD": (
                "{subject}_V1_MR/MNINonLinear/"
                "Results/{task}_{phase_encoding}/"
                "{task}_{phase_encoding}_hp0_clean.nii.gz"
            )
        }
        types = list(patterns.keys())

        # Set default tasks
        if tasks is None:
            self.tasks: List[str] = all_tasks
        # Convert single task into list
        else:
            if not isinstance(tasks, List):
                tasks = [tasks]

            # Check for invalid task(s)
            for task in tasks:
                if task not in all_tasks:
                    raise_error(
                        f"'{task}' is not a valid HCP-Aging fMRI task input. "
                        f"Valid task values can be any or all of {all_tasks}."
                    )
            self.tasks: List[str] = tasks

        # All phase encodings
        all_phase_encodings = ["AP", "PA"]

        # Set phase encodings
        if phase_encodings is None:
            phase_encodings = all_phase_encodings

        # Convert single phase encoding into list
        if isinstance(phase_encodings, str):
            phase_encodings = [phase_encodings]

        # Check for invalid phase encoding(s)
        for pe in phase_encodings:
            if pe not in all_phase_encodings:
                raise_error(
                    f"'{pe}' is not a valid HCP-Aging phase encoding. "
                    "Valid phase encoding can be any or all of "
                    f"{all_phase_encodings}."
                )

        self.phase_encodings = phase_encodings

        # The replacements
        replacements = ["subject", "task", "phase_encoding"]
        super().__init__(
            types=types,
            datadir=datadir,
            patterns=patterns,
            replacements=replacements,
            confounds_format="adhoc",
        )

    def get_item(self, subject: str, task: str, phase_encoding: str) -> Dict:
        """Index one element in the dataset.

        Parameters
        ----------
        subject : str
            The subject ID.
        tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"}
            The task.
        phase_encoding : {"PA", "AP"}
            The phase encoding.

        Returns
        -------
        out : dict
            Dictionary of paths for each type of data required for the
            specified element.
        """
        # Resting task
        if "REST" in task:
            new_task = f"rfMRI_{task}"
        else:
            new_task = f"tfMRI_{task}"

        out = super().get_item(
            subject=subject, task=new_task, phase_encoding=phase_encoding
        )
        return out

    def get_elements(self) -> List:
        """Implement fetching list of elements in the dataset.

        Returns
        -------
        list
            The list of elements in the dataset.
        """
        subjects = [
            x.name.split("_")[0]
            for x in self.datadir.iterdir()
            if "V1_MR" in x.name
        ]

        return [
            (sub, task, phase_encoding)
            for sub, task, phase_encoding in product(
                subjects, self.tasks, self.phase_encodings
            )
        ]

    @property
    def skip_file_check(self) -> bool:
        """Skip file check existence."""
        return True

@register_datagrabber
class DataladHCPAging(DataladDataGrabber, HCPAging):
    """Concrete implementation for datalad-based HCP Aging dataset.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2", "CARIT", "FACENAME", "VISMOTOR"}
        or list of the options, optional
        HCP aging task sessions. If None, all available task sessions
        are selected (default None).
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP aging phase encoding directions. If None, both will be used
        (default None).
    **kwargs
        Keyword arguments passed to superclass.

    """

    def __init__(
        self,
        datadir: Union[str, Path, None] = None,
        tasks: Union[str, List[str], None] = None,
        phase_encodings: Union[str, List[str], None] = None,
    ) -> None:
        """Initialise the class."""
        uri = (
            "ria+http://hcp-a.ds.inm7.de"
            "#431e4eb7-72c3-423d-b402-84ede40c03a2"
        )

        super().__init__(
            datadir=datadir,
            tasks=tasks,
            phase_encodings=phase_encodings,
            uri=uri,
        )


if __name__ == "__main__":
    with DataladHCPAging() as dg:
        elems = dg.get_elements()
        test_data = dg[elems[0]]
        print(test_data)