"""Provide base class for InstantPhaseConnectivity marker."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
# License: AGPL

from abc import abstractmethod
from typing import Any, ClassVar, Optional, Union

import numpy as np
from nilearn import signal as nil_signal
from scipy import signal, stats

from junifer.markers import BaseMarker
from junifer.typing import Dependencies, MarkerInOutMappings
from junifer.utils import logger, raise_error


__all__ = ["InstantPhaseConnectivityBase"]


class InstantPhaseConnectivityBase(BaseMarker):
    """Abstract base class for instantaneous phase coherence connectivity.

    Parameters
    ----------
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

    _DEPENDENCIES: ClassVar[Dependencies] = {"nilearn", "scipy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[MarkerInOutMappings] = {
        "BOLD": {
            "fc": "timeseries_2d",
        },
    }

    def __init__(
        self,
        highpass: float,
        lowpass: float,
        order: int = 5,
        masks: Union[str, dict, list[Union[dict, str]], None] = None,
        tr: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        if highpass < 0:
            raise_error("Highpass must be positive or 0")
        if lowpass <= 0:
            raise_error("Lowpass must be positive")
        if order <= 0:
            raise_error("Order must be positive")
        if highpass >= lowpass:
            raise_error("Highpass must be lower than lowpass")
        self.highpass = highpass
        self.lowpass = lowpass
        self.order = order
        self.tr = tr
        self.masks = masks
        super().__init__(on="BOLD", name=name)

    @abstractmethod
    def aggregate(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform aggregation.

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
        raise_error(
            msg="Concrete classes need to implement aggregate().",
            klass=NotImplementedError,
        )

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict] = None,
    ) -> dict:
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
            The computed result as dictionary. This will be either returned
            to the user or stored in the storage by calling the store method
            with this as a parameter. The dictionary has the following keys:

            * ``aggrefcgation`` : dictionary with the following keys:

                - ``data`` : ROI values as ``numpy.ndarray``
                - ``col_names`` : ROI labels as list of str
                - ``row_names`` : ROI labels as list of str
        Warns
        -----
        RuntimeWarning
            If time aggregation is required but only time point is available.

        """

        aggregation = self.aggregate(input, extra_input=extra_input)

        roi_data = aggregation["aggregation"]["data"]  # n_timepoints x n_rois

        # Z-scored data
        zscore_roi_data = stats.zscore(roi_data, axis=0)  # z-score across time

        tr = self.tr
        if tr is None:
            tr = float(input["data"].header.get_zooms()[3])  # type: ignore
            logger.info(f"`tr` not provided, using `tr` from header: {tr}")

        # Filter
        filtered_data = nil_signal.butterworth(
            zscore_roi_data,
            sampling_rate=1 / tr,
            low_pass=self.lowpass,
            high_pass=self.highpass,
            order=self.order,
            copy=True,
        )

        # Hilbert transform
        phi = np.angle(signal.hilbert(filtered_data, axis=0))

        # phi is n_timepoints x n_rois
        # Cosine distance
        iphc = np.cos(phi[:, None, :] - phi[:, :, None])

        out = {
            "fc": {
                "data": iphc,
                "col_names": aggregation["aggregation"]["col_names"],
                "row_names": aggregation["aggregation"]["col_names"],
            },
        }
        return out
