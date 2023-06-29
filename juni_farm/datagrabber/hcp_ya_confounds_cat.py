"""Provide a datagrabber for CAT-processed confounds in the HCP dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from itertools import product
from pathlib import Path
from typing import Dict, List, Union

from junifer.api.decorators import register_datagrabber
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
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options, optional
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
            "SOCIAL",
            "WM",
            "RELATIONAL",
            "EMOTION",
            "LANGUAGE",
            "GAMBLING",
            "MOTOR",
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
                        f"'{task}' is not a valid HCP-YA fMRI task input. "
                        f"Valid task values can be any or all of {all_tasks}."
                    )
            self.tasks: List[str] = tasks

        # All phase encodings
        all_phase_encodings = ["LR", "RL"]
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
                    f"'{pe}' is not a valid HCP-YA phase encoding. "
                    "Valid phase encoding can be any or all of "
                    f"{all_phase_encodings}."
                )

        # The types of data
        types = ["BOLD_confounds"]
        # The patterns
        patterns = {
            "BOLD_confounds": (
                "sub-{subject}/sub-{subject}_task-{task}"
                "{phase_encoding}_desc-confounds_timeseries.tsv"
            )
        }
        # The replacements
        replacements = [
            "subject",
            "task",
            "phase_encoding",
        ]
        uri = (
            "ria+file:///data/project/"
            "cat_preprocessed/dataladstore#~HCP-YA_conf"
        )
        super().__init__(
            types=types,
            datadir=datadir,
            uri=uri,
            patterns=patterns,
            replacements=replacements,
            confounds_format="adhoc",
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

        out = super().get_item(
            subject=subject, task=new_task, phase_encoding=new_phase_encoding
        )
        out["BOLD_confounds"]["mappings"] = {
            "fmriprep": get_cat_to_fmriprep_mapping(),
        }
        return out

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
class JuselessDataladHCP1200(DataladDataGrabber, HCP1200):
    """Concrete implementation for datalad-based data fetching of HCP1200.

    This implementation uses a git repository on jusless, located at
    ``/data/group/appliedml``.

    Parameters
    ----------
    datadir : str or Path or None, optional
        The directory where the datalad dataset will be cloned. If None,
        the datalad dataset will be cloned into a temporary directory
        (default None).
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options or None \
            , optional
        HCP task sessions. If None, all available task sessions are selected
        (default None).
    phase_encodings : {"LR", "RL"} or list of the options or None, optional
        HCP phase encoding directions. If None, both will be used
        (default None).
    ica_fix : bool, optional
        Whether to retrieve data that was processed with ICA+FIX.
        Only "REST1" and "REST2" tasks are available with ICA+FIX (default
        False).

    """

    def __init__(
        self,
        datadir: Union[str, Path, None] = None,
        tasks: Union[str, List[str], None] = None,
        phase_encodings: Union[str, List[str], None] = None,
        ica_fix: bool = False,
    ) -> None:
        uri = (
            "/data/group/appliedml/datalad-datasets/"
            "human-connectome-project-openaccess.git"
        )
        rootdir = "HCP1200"
        super().__init__(
            datadir=datadir,
            tasks=tasks,
            phase_encodings=phase_encodings,
            uri=uri,
            rootdir=rootdir,
            ica_fix=ica_fix,
        )

    # Needed here as HCP1200's subjects are sub-datasets, so will not be
    # found when elements are checked.
    @property
    def skip_file_check(self) -> bool:
        """Skip file check existence."""
        return True



@register_datagrabber
class MultipleHCP(MultipleDataGrabber):
    """Concrete implementation for original HCP data confounds.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2", "SOCIAL", "WM", "RELATIONAL", "EMOTION", \
            "LANGUAGE", "GAMBLING", "MOTOR"} or list of the options, optional
        HCP task sessions. If None, all available task sessions are selected
        (default None).
    phase_encodings : {"LR", "RL"} or list of the options, optional
        HCP phase encoding directions. If None, both will be used
        (default None).
    **kwargs
        Keyword arguments passed to superclass.

    """

    def __init__(self, **kwargs):
        """Initialise class."""
        super().__init__(
            datagrabbers=[
                JuselessDataladHCP1200(**kwargs),
                HCPCATConfounds(**kwargs)],
            **kwargs,
        )


# test
if __name__ == "__main__":
    with MultipleHCP(tasks="REST1") as hcp_conf:
        all_elements = hcp_conf.get_elements()

        # this will run over all elements, so use keyboard interrupt
        # if convinced that it works
        for element in all_elements:
            print(element)
            out = hcp_conf[element]
            print(out)
