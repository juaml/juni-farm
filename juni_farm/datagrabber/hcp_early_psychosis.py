"""Provide a datagrabber for the HCP Early Psychosis dataset."""

# Authors: Leonard Sasse <l.sasse@fz-juelich.de>
# License: AGPL

from itertools import product
from pathlib import Path
from typing import Dict, List, Union

from junifer.datagrabber.datalad_base import DataladDataGrabber
from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber
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
class HCPEarlyPsychosis(PatternDataGrabber):
    """Concrete implementation for HCP Early Psychosis dataset.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2"}
        or list of the options, optional
        HCP early psychosis task sessions. If None, all available task sessions
        are selected (default None).
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP early psychosis phase encoding directions. If None, both will be
        used (default None).
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
        all_tasks = ["REST1", "REST2"]
        patterns = {
            "BOLD": (
                "{subject}_01_MR/MNINonLinear/"
                "Results/rfMRI_{task}_{phase_encoding}/"
                "rfMRI_{task}_{phase_encoding}_hp0_clean.nii.gz"
            ),
            "BOLD_confounds": (
                "HCP_EP_confounds/{subject}_01_MR/MNINonLinear/"
                "Results/rfMRI_{task}_{phase_encoding}/"
                "Confounds_{subject}_01_MR.tsv"
            ),
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
                        f"'{task}' is not a valid HCP Early Psychosis"
                        " fMRI task input. "
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
                    f"'{pe}' is not a valid HCP Early Psychosis "
                    "phase encoding. Valid phase encoding can be"
                    f" any or all of {all_phase_encodings}."
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
        task : {"REST1", "REST2"}
            The task.
        phase_encoding : {"AP", "PA"}
            The phase encoding.

        Returns
        -------
        out : dict
            Dictionary of paths for each type of data required for the
            specified element.
        """
        out = super().get_item(
            subject=subject, task=task, phase_encoding=phase_encoding
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
        subjects = [
            x.name.split("_")[0]
            for x in self.datadir.iterdir()
            if "01_MR" in x.name
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
class DataladHCPEarlyPsychosis(DataladDataGrabber, HCPEarlyPsychosis):
    """Concrete implementation for datalad-based HCP Early Psychosis dataset.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    tasks : {"REST1", "REST2"}
        or list of the options, optional
        HCP Early Psychosis task sessions. If None, all available task sessions
        are selected (default None).
    phase_encodings : {"AP", "PA"} or list of the options, optional
        HCP Early Psychosis phase encoding directions. If None, both will be
        used (default None).
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
        # there is also a jugit dataset that we can use, i currently dont have
        # access rights to it, so i am using the path on juseless
        # uri = "git@jugit.fz-juelich.de:inm7/hcp_earlypsychosis_preprocessed"
        uri = "/data/project/hcp_earlypsychosis_preprocessing"
        super().__init__(
            datadir=datadir,
            tasks=tasks,
            phase_encodings=phase_encodings,
            uri=uri,
        )


if __name__ == "__main__":
    with DataladHCPEarlyPsychosis() as dg:
        elems = dg.get_elements()
        print(elems)
        test_data = dg[elems[0]]
        print(test_data)
