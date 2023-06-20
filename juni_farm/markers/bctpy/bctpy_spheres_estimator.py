"""Provide estimator class for BCTPY"""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, List

from junifer.markers import FunctionalConnectivitySpheres
from junifer.markers.utils import singleton
from junifer.utils import logger, raise_error

if TYPE_CHECKING:
    from nibabel import Nifti1Image, Nifti2Image


@singleton
class BCTPYSpheresEstimator:
    """Estimator class for BCTPY using spheres.

    This class is a singleton and is used for efficient computation of BCTPY
    metrics, by caching the results of the FC computation.

    .. warning:: This class can only be used via :class:`.BCTPYSpheresBase` as
    it serves a specific purpose.

    """

    def __init__(self) -> None:
        self._file_path = None

    @lru_cache(maxsize=None, typed=True)  # noqa: B019
    def _compute(
        coords: str,
        radius: Optional[float] = None,
        allow_overlap: bool = False,
        agg_method: str = "mean",
        agg_method_params: Optional[Dict] = None,
        cor_method: str = "covariance",
        cor_method_params: Optional[Dict] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
    ) -> Dict[str, Any]:
        estimator = FunctionalConnectivitySpheres(
            coords=coords,
            radius=radius,
            allow_overlap=allow_overlap,
            agg_method=agg_method,
            agg_method_params=agg_method_params,
            cor_method=cor_method,
            cor_method_params=cor_method_params,
            masks=masks,
        )

        return estimator.fit_transform()

    def fit_transform(
        self,
        input_data: Dict[str, Any],
        coords: str,
        radius: Optional[float] = None,
        allow_overlap: bool = False,
        agg_method: str = "mean",
        agg_method_params: Optional[Dict] = None,
        cor_method: str = "covariance",
        cor_method_params: Optional[Dict] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
    ) -> Dict[str, Any]:
        """Fit and transform for the estimator.

        Parameters
        ----------
        coords : str
            The name of the coordinates list to use. See
            :func:`.list_coordinates` for options.
        radius : float, optional
            The radius of the sphere in mm. If None, the signal will be extracted
            from a single voxel. See :class:`nilearn.maskers.NiftiSpheresMasker`
            for more information (default None).
        allow_overlap : bool, optional
            Whether to allow overlapping spheres. If False, an error is raised if
            the spheres overlap (default is False).
        agg_method : str, optional
            The aggregation method to use.
            See :func:`.get_aggfunc_by_name` for more information
            (default None).
        agg_method_params : dict, optional
            The parameters to pass to the aggregation method (default None).
        cor_method : str, optional
            The method to perform correlation using. Check valid options in
            :class:`nilearn.connectome.ConnectivityMeasure` (default "covariance").
        cor_method_params : dict, optional
            Parameters to pass to the correlation function. Check valid options in
            :class:`nilearn.connectome.ConnectivityMeasure` (default None).
        masks : str, dict or list of dict or str, optional
            The specification of the masks to apply to regions before extracting
            signals. Check :ref:`Using Masks <using_masks>` for more details.
            If None, will not apply any mask (default None).

        Returns
        -------
        Dict[str, Any]
            The computed FC metric.
        """
        bold_path = input_data["path"]
        bold_data = input_data["data"]
        # Clear cache if file path is different from when caching was done
        if self._file_path != bold_path:
            logger.info(f"Removing fALFF map cache at {self._file_path}.")
            # Clear the cache
            self._compute.cache_clear()
            # Set the new file path
            self._file_path = bold_path
        else:
            logger.info(f"Using fALFF map cache at {self._file_path}.")
        # Compute
        return self._compute(
            data=bold_data,
            coords=coords,
            radius=radius,
            allow_overlap=allow_overlap,
            agg_method=agg_method,
            agg_method_params=agg_method_params,
            cor_method=cor_method,
            cor_method_params=cor_method_params,
            masks=masks,
        )
