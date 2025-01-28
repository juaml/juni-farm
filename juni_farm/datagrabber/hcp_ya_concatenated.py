from typing import Dict, List

import nibabel as nib
import pandas as pd
from .hcp_ya_confounds_cat import MultipleHCP

from junifer.api.decorators import register_datagrabber
from junifer.pipeline import WorkDirManager
from junifer.utils import logger


def junifer_module_deps() -> List[str]:
    """Return the dependencies of the module.

    Returns
    -------
    List[str]
        The list of dependencies.

    """
    return ["hcp_ya_confounds_cat.py"]


@register_datagrabber
class HCP_YA_Concatenated(MultipleHCP):
    """Concatenate all tasks and phase encoding directions for the HCP YA.

    Parameters
    ----------
    datadir : str or Path, optional
        The directory where the datalad dataset will be cloned.
    **kwargs
        Keyword arguments passed to superclass.

    """

    def __init__(self, **kwargs):
        super().__init__(
            tasks=None, phase_encodings=None, ica_fix=False, **kwargs
        )

    def get_elements(self) -> List[str]:
        all_elements = super().get_elements()
        return list({element[0] for element in all_elements})

    def __getitem__(self, subject: str) -> Dict:
        all_data = []
        all_tasks = [
            "SOCIAL",
            "WM",
            "RELATIONAL",
            "EMOTION",
            "LANGUAGE",
            "GAMBLING",
            "MOTOR",
        ]
        all_pe = ["LR", "RL"]
        for task in all_tasks:
            for pe in all_pe:
                all_data.append(super().__getitem__((subject, task, pe)))

        all_bolds = []
        all_confounds = []
        for data in all_data:
            all_bolds.append(data["BOLD"]["path"])
            all_confounds.append(data["BOLD"]["confounds"]["path"])

        logger.info("Concatenating BOLD images and confounds")
        concat_img = nib.concat_images(all_bolds, axis=3)
        concat_confounds = pd.concat(
            [pd.read_csv(conf, sep="\t") for conf in all_confounds]
        )
        logger.info("Saving concatenated BOLD images and confounds")
        tmpdir = WorkDirManager().get_element_tempdir(prefix="hcp_ya_concat")
        concat_bold_fname = tmpdir / "concat_bold.nii.gz"
        concat_confounds_fname = tmpdir / "concat_confounds.tsv"
        nib.save(concat_img, concat_bold_fname)
        concat_confounds.to_csv(concat_confounds_fname, sep="\t", index=False)

        new_data = all_data[0].copy()
        new_data["BOLD"]["path"] = concat_bold_fname
        new_data["BOLD"]["confounds"]["path"] = concat_confounds_fname
        new_data["BOLD"]["meta"]["element"] = {"subject": subject}
        return new_data


if __name__ == "__main__":
    from junifer.utils import configure_logging

    configure_logging(level="INFO")
    with HCP_YA_Concatenated() as hcp_concat:
        all_elements = hcp_concat.get_elements()

        # this will run over all elements, so use keyboard interrupt
        # if convinced that it works
        for element in all_elements[:1]:
            print(element)

        element = "100307"
        out = hcp_concat[element]
        print(out)
