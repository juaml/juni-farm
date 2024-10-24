"""Provide a datagrabber for the HCP Aging dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>, Jean-Philippe Kroell <j.kroell@fz-juelich.de>
# License: AGPL

from itertools import product
from pathlib import Path
from typing import Dict, List, Union

from junifer.datagrabber.datalad_base import DataladDataGrabber
from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber
from junifer.utils import raise_error


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
