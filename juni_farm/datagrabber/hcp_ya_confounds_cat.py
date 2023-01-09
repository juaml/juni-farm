"""Provide a datagrabber for CAT-processed confounds in the HCP dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from pathlib import Path
from typing import Dict, List, Union

from junifer.utils import raise_error
from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataladDataGrabber


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

    mapping = {"FD": "framewise_displacement"}

    for cat, fmriprep in zip(terms_cat, terms_fmriprep):
        mapping[cat] = fmriprep
        mapping[f"{cat}^2"] = f"{fmriprep}_power2"

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
            new_task = f"rfMRI_{task}"
            new_phase_encoding = f"{phase_encoding}hp2000clean"
        else:
            new_task = f"tfMRI_{task}"
            new_phase_encoding = phase_encoding

        out = super().get_item(
            subject=subject, task=new_task, phase_encoding=new_phase_encoding
        )
        return out


# test
if __name__ == "__main__":

    bold_confounds = {
        "BOLD_confounds": {
            "path": (
                "sub-{subject}/sub-{subject}_task-{task}"
                "{phase_encoding}_desc-confounds_timeseries.tsv"
            ),
            "format": "adhoc",
            "mappings": {
                "fmriprep": get_cat_to_fmriprep_mapping(),
            },
        }
    }
    with HCPCATConfounds() as hcp_conf:
        all_elements = hcp_conf.get_elements()

        # this will run over all elements, so use keyboard interrupt
        # if convinced that it works
        for element in all_elements:
            print(element)
            out = hcp_conf[element]
            out["BOLD_confounds"] = bold_confounds
            print(out)
