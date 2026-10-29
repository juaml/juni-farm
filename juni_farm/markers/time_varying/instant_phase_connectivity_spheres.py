"""Provide class for instant phase connectivity using spheres."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Literal, Optional

from pydantic import PositiveFloat

from junifer.api.decorators import register_marker
from junifer.datagrabber import DataType
from junifer.markers import SphereAggregation

from .instant_phase_connectivity_base import InstantPhaseConnectivityBase


__all__ = ["InstantPhaseConnectivitySpheres"]


@register_marker
class InstantPhaseConnectivitySpheres(InstantPhaseConnectivityBase):
    """Class for instantaneous phase coherence connectivity using spheres.

    Parameters
    ----------
    coords : str
        The name of the coordinates list to use.
        See :func:`.list_data` for options.
    radius : ``zero`` or positive float or None, optional
        The radius of the sphere in millimetres.
        If None, the signal will be extracted from a single voxel.
        See :class:`.JuniferNiftiSpheresMasker` for more information
        (default None).
    allow_overlap : bool, optional
        Whether to allow overlapping spheres.
        If False, an error is raised if the spheres overlap (default False).
    highpass : positive float
        Highpass cutoff frequency.
    lowpass : positive float
        Lowpass cutoff frequency.
    order : positive int, optional
        The order of the Butterworth filter (default 5).
    masks : str, dict or list of dict or str, optional
        The specification of the masks to apply to regions before extracting
        signals. Check :ref:`Using Masks <using_masks>` for more details.
        If None, will not apply any mask (default None).
    tr : positive float, optional
        The Repetition Time of the BOLD data. If None, will extract
        the TR from NIfTI header (default None).
    name : str, optional
        The name of the marker. If None, it will use the class name
        (default None).

    Notes
    -----
    The ``tr`` parameter is crucial for the correctness of the filter
    computation. If a dataset is correctly preprocessed, the ``tr`` should be
    extracted from the NIfTI without any issue. However, it has been
    reported that some preprocessed data might not have the correct ``tr`` in
    the NIfTI header.

    Raises
    ------
    ValueError
        if ``highpass`` is higher than ``lowpass``.

    """

    coords: str
    radius: Literal[0] | PositiveFloat | None = None
    allow_overlap: bool = False

    def aggregate(
        self, input: dict[str, Any], extra_input: Optional[dict] = None
    ) -> dict:
        """Perform sphere aggregation.

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
            The computed result as dictionary. This will be either returned
            to the user or stored in the storage by calling the store method
            with this as a parameter. The dictionary has the following keys:

            * ``aggregation`` : dictionary with the following keys:

                - ``data`` : ROI values as ``numpy.ndarray``
                - ``col_names`` : ROI labels as list of str

        """
        return SphereAggregation(
            coords=self.coords,
            radius=self.radius,
            allow_overlap=self.allow_overlap,
            method="mean",
            masks=self.masks,
            on=DataType.BOLD,
        ).compute(input=input, extra_input=extra_input)
