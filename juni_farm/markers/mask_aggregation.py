"""Provide class for mask aggregation."""

# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Agustina Aragon Daud <a.aragon@fz-juelich.de>
# License: AGPL

from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from nilearn.maskers import NiftiMasker
from pydantic import BeforeValidator

from junifer.api.decorators import register_marker
from junifer.data import get_data
from junifer.datagrabber import DataType
from junifer.markers.base import BaseMarker, logger
from junifer.stats import get_aggfunc_by_name
from junifer.storage import StorageType
from junifer.typing import Dependencies, MarkerInOutMappings
from junifer.utils import ensure_list_or_none, raise_error, warn_with_log


__all__ = ["MaskAggregation"]

_on = Literal[
    DataType.T1w,
    DataType.T2w,
    DataType.BOLD,
    DataType.VBM_GM,
    DataType.VBM_WM,
    DataType.VBM_CSF,
    DataType.FALFF,
    DataType.GCOR,
    DataType.LCOR,
]


@register_marker
class MaskAggregation(BaseMarker):
    """Class for mask aggregation.

    Aggregates the values of a given data type within a mask using a specified
    aggregation function. If the resulting aggregation is a time series
    (e.g., for BOLD data), an optional time aggregation function can be applied
    to further reduce the time dimension. If the time aggregation function is 
    not specified, the marker will compute the derivative, power, and power of
    the derivative of the time series, matching the idea of the derivatives
    and power of the motion parameters computed by fMRIPrep.

    Parameters
    ----------
    method : str, optional
        The aggregation function to use.
        See :func:`.get_aggfunc_by_name` for options
        (default "mean").
    method_params : dict or None, optional
        The parameters to pass to the aggregation function.
        See :func:`.get_aggfunc_by_name` for options (default None).
    time_method : str or None, optional
        The aggregation function to use for time series after applying
        :term:`method` (only applicable to BOLD data). If None,
        it will not operate on the time dimension (default None).
    time_method_params : dict or None, optional
        The parameters to pass to the time aggregation function (default None).
    on : {``DataType.T1w``, ``DataType.T2w``, ``DataType.BOLD``, \
         ``DataType.VBM_GM``, ``DataType.VBM_WM``, ``DataType.VBM_CSF``, \
         ``DataType.FALFF``, ``DataType.GCOR``, ``DataType.LCOR``} or \
         list of them or None, optional
        The data type(s) to apply the marker on.
        If None, will work on BOLD  data.
        Check :enum:`.DataType` for valid values (default None).
    masks : str, dict, list of them or None, optional
        The specification of the masks to apply to regions before extracting
        signals. Check :ref:`Using Masks <using_masks>` for more details.
        If None, will not apply any mask (default None).
    name : str or None, optional
        The name of the marker.
        If None, will use the class name (default None).

    Raises
    ------
    ValueError
        If ``time_method`` is specified for non-BOLD data or if
        ``time_method_params`` is not None when ``time_method`` is None.

    """

    _DEPENDENCIES: ClassVar[Dependencies] = {"nilearn", "numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[MarkerInOutMappings] = {
        DataType.T1w: {
            "aggregation": StorageType.Vector,
        },
        DataType.T2w: {
            "aggregation": StorageType.Vector,
        },
        DataType.BOLD: {
            "aggregation": StorageType.Timeseries,
        },
        DataType.VBM_GM: {
            "aggregation": StorageType.Vector,
        },
        DataType.VBM_WM: {
            "aggregation": StorageType.Vector,
        },
        DataType.VBM_CSF: {
            "aggregation": StorageType.Vector,
        },
        DataType.FALFF: {
            "aggregation": StorageType.Vector,
        },
        DataType.GCOR: {
            "aggregation": StorageType.Vector,
        },
        DataType.LCOR: {
            "aggregation": StorageType.Vector,
        },
    }

    method: str = "mean"
    method_params: dict[str, Any] | None = None
    time_method: str | None = None
    time_method_params: dict[str, Any] | None = None
    masks: Annotated[
        dict | str | list[dict | str] | None,
        BeforeValidator(ensure_list_or_none),
    ] = None
    on: Annotated[
        _on | list[_on] | None, BeforeValidator(ensure_list_or_none)
    ] = None

    def validate_marker_params(self) -> None:
        """Run extra logical validation for marker."""
        # self.on is set already
        if "BOLD" not in self.on and self.time_method is not None:
            raise_error(
                "`time_method` can only be used with BOLD data. "
                "Please remove `time_method` parameter."
            )
        if self.time_method is None and self.time_method_params is not None:
            raise_error(
                "`time_method_params` can only be used with `time_method`. "
                "Please remove `time_method_params` parameter."
            )

    def compute(
        self, input: dict[str, Any], extra_input: dict | None = None
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

            * ``aggregation`` : dictionary with the following keys:

                - ``data`` : values as ``numpy.ndarray``
                - ``col_names`` : metric labels as list of str

        """
        t_input_img = input["data"]
        logger.debug(f"Whole-mask aggregation using {self.method}")
        # Get aggregation function
        agg_func = get_aggfunc_by_name(
            name=self.method, func_params=self.method_params
        )

        # Load mask
        logger.debug(f"Masking with {self.masks}")
        mask_img = get_data(
            kind="mask",
            names=self.masks,
            target_data=input,
            extra_input=extra_input,
        )

        # Initialize masker
        logger.debug("Masking")
        masker = NiftiMasker(mask_img, target_affine=t_input_img.affine)
        # Mask the input data
        data = masker.fit_transform(t_input_img)


        # Apply aggregation function across all voxels in the mask
        logger.debug("Computing mask means")
        mask_agg = agg_func(data, axis=-1)

        # Store output values and column names
        col_names = [self.method]
        out_values = [mask_agg]

        # Apply time dimension aggregation if required
        if self.time_method is not None:
            if mask_agg.shape[0] > 1:
                logger.debug("Aggregating time dimension")
                time_agg_func = get_aggfunc_by_name(
                    self.time_method, func_params=self.time_method_params
                )
                mask_agg = time_agg_func(mask_agg, axis=0)
            else:
                warn_with_log(
                    "No time dimension to aggregate as only one time point is "
                    "available."
                )
        else:
            if mask_agg.shape[0] > 1:
                # Compute derivative
                logger.debug("Computing derivative")
                derivative = np.insert(np.diff(mask_agg), 0, np.nan)
                col_names.append("derivative")
                out_values.append(derivative)

                # Compute power
                logger.debug("Computing power")
                power = mask_agg**2
                col_names.append("power")
                out_values.append(power)

                # Computer power of derivative
                logger.debug("Computing power of derivative")
                derivative_power = derivative**2
                col_names.append("derivative_power")
                out_values.append(derivative_power)

        # Format the output
        out_values = np.column_stack(out_values)
        return {
            "aggregation": {
                "data": out_values,
                "col_names": col_names,
            },
        }
