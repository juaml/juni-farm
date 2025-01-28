"""Provide class for instant phase connectivity using parcels."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from typing import Any, Optional, Union

from junifer.api.decorators import register_marker
from junifer.markers import ParcelAggregation

from .instant_phase_connectivity_base import InstantPhaseConnectivityBase


__all__ = ["InstantPhaseConnectivityParcels"]


@register_marker
class InstantPhaseConnectivityParcels(InstantPhaseConnectivityBase):
    """Class for instantaneous phase coherence connectivity using parcels.

    Parameters
    ----------
    parcellation : str or list of str
        The name(s) of the parcellation(s) to use.
        See :func:`.list_data` for options.
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

    -----
    The ``tr`` parameter is crucial for the correctness of the filter
    computation. If a dataset is correctly preprocessed, the ``tr`` should be
    extracted from the NIfTI without any issue. However, it has been
    reported that some preprocessed data might not have the correct ``tr`` in
    the NIfTI header.

    Raises
    ------
    ValueError
        If ``highpass`` is not positive or zero or
        if ``lowpass`` is not positive or
        if ``highpass`` is higher than ``lowpass`` or

    """

    def __init__(
        self,
        parcellation: Union[str, list[str]],
        highpass: float,
        lowpass: float,
        order: int = 5,
        masks: Union[str, dict, list[Union[dict, str]], None] = None,
        tr: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        self.parcellation = parcellation
        super().__init__(
            highpass=highpass,
            lowpass=lowpass,
            order=order,
            masks=masks,
            tr=tr,
            name=name,
        )

    def aggregate(
        self, input: dict[str, Any], extra_input: Optional[dict] = None
    ) -> dict:
        """Perform parcel aggregation.

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
        aggregation = ParcelAggregation(
            parcellation=self.parcellation,
            method="mean",
            masks=self.masks,
            on="BOLD",
        ).compute(input=input, extra_input=extra_input)
        return aggregation
