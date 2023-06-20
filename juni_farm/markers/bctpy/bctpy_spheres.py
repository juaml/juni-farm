"""Provide estimator class for BCTPY"""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
#          Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, List

from junifer.api import register_marker
from junifer.markers import FunctionalConnectivitySpheres
from junifer.utils import logger, raise_error

if TYPE_CHECKING:
    from nibabel import Nifti1Image, Nifti2Image

import bct

_metrics_dict = {
    "degrees_und": bct.algorithms.degrees_und,
}


@register_marker
class BCTPYSpheres(FunctionalConnectivitySpheres):
    """Class to compute BCTPY markers using spheres.

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
    name : str, optional
        The name of the marker. By default, it will use
        KIND_FunctionalConnectivitySpheres where KIND is the kind of data it
        was applied to (default None).


    """

    def __init__(
        self,
        metrics: Union[str, List[str]],
        coords: str,
        radius: Optional[float] = None,
        allow_overlap: bool = False,
        agg_method: str = "mean",
        agg_method_params: Optional[Dict] = None,
        cor_method: str = "covariance",
        cor_method_params: Optional[Dict] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics
        super().__init__(
            coords=coords,
            radius=radius,
            allow_overlap=allow_overlap,
            agg_method=agg_method,
            agg_method_params=agg_method_params,
            cor_method=cor_method,
            cor_method_params=cor_method_params,
            masks=masks,
            name=name,
        )

    def get_output_type(self, input_type: str) -> str:
        """Get output type.

        Parameters
        ----------
        input_type : str
            The data type input to the marker.

        Returns
        -------
        str
            The storage type output by the marker.

        """
        return "vector"

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict] = None,
    ) -> Dict:
        """Compute.

        Parameters
        ----------
        input : dict
            A single input from the pipeline data object in which to compute
            the marker.
        extra_input : dict, optional
            The other fields in the pipeline data object. Useful for accessing
            other data kind that needs to be used in the computation. For
            example, the functional connectivity markers can make use of the
            confounds if available (default None).

        Returns
        -------
        dict
            The computed result as dictionary. The following keys will be
            included in the dictionary:

            * ``data`` : functional connectivity matrix as a ``numpy.ndarray``.
            * ``row_names`` : row names as a list
            * ``col_names`` : column names as a list
            * ``matrix_kind`` : the kind of matrix (tril, triu or full)

        """
        logger.info("Computing FC for BCTPY")
        conn_out = super().compute(input, extra_input)
        # TODO: Compute the BCTPY markers
        logger.info("Computing BCTPY markers")

        graph = conn_out["data"]

        out_vals = []
        for t_metric in self.metrics:
            metric = _metrics_dict[t_metric]
            out_vals.append(metric(graph))

        out = {
            "data": out_vals,
            "col_names": self.metrics,
        }
        return out