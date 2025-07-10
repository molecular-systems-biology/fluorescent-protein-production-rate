from __future__ import annotations

import copy
import hashlib
import inspect
from typing import (
    Any, Self, Dict, Tuple, List, Optional, Iterator, Sequence, Callable
)
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, RationalQuadratic, ConstantKernel
)

# This should match with the version string in setup.py and the GitHub
# release tag.
_VERSION: str = "1.0"


class MissingDataWarning(Warning):
    """Warns when non-essential data has missing values."""
    pass


class InsufficientDataWarning(Warning):
    """Warns when available data is sufficient, but less than ideal."""
    pass


class CellCycle:
    """
    Represents a single cell cycle. Provides methods for analyzing
    fluorescent protein dynamics from raw volume and concentration data 
    through to volume-specific protein production rates.

    Notes
    -----
    The main analysis methods need to be called in the following order:
    1. `merge_cycle_data()`: Combines mother cell and bud data into a 
        unified time series.
    2. `calculate_abundance()`: Computes total protein abundance from
        concentration and volume.
    3. `calculate_smooth_abundance()`: Applies Gaussian process 
        smoothing.
    4. `calculate_production_rate()`: Calculates rate of change of
        abundance, optionally applying maturation correction.
    5. `calculate_smoothed_volume()`: Applies Gaussian process
        smoothing to volume data.
    6. `calculate_volume_specific_production_rate()`: Computes
        volume-specific production rates.

    The convenience method `calculate_all_cycle_values()` runs all of 
    these steps in the correct order, allowing users to process
    data in one go.
    """
    def __init__(
        self, 
        cycle_id: str,
        mother_data: pd.DataFrame,
        previous_bud_data: pd.DataFrame,
        current_bud_data: pd.DataFrame,
        cycle_events: Dict[str, int],
        cycle_end_event: str,
        min_extra_data_points: int = 3,
        max_extra_data_points: int = 8
    ) -> None:
        """
        Initialize a CellCycle with experimental data.

        Parameters
        ----------
        cycle_id : str
            A unique identifier for the cell cycle.
        mother_data : pd.DataFrame
            DataFrame containing data for the mother cell. Requires at
            least integer TimeID, float Volume, float Concentration and
            boolean Interpolate columns.
        previous_bud_data : pd.DataFrame
            DataFrame containing data for the previous bud. Requires at
            least integer TimeID, float Volume and boolean Interpolate
            columns.
        current_bud_data : pd.DataFrame
            DataFrame containing data for the current bud. Requires at
            least integer TimeID, float Volume and boolean Interpolate
            columns.
        cycle_events : Dict[str, int]
            A dictionary mapping event names to their corresponding time
            points. Requires at least Bud_0 and Bud_1 key value pairs as
            well as pairs for the cycle end events.
        cycle_end_event : str
            The name of the event that marks the end of the cell cycle.
            For example, if cycle ends are defined by "Mitotic_exit_0"
            and "Mitotic_exit_1", this should be "Mitotic_exit".
        min_extra_data_points : int, optional
            Minimum number of extra data points before and after the 
            cycle's beginning and end for smoothing. Default is 3.
        max_extra_data_points : int, optional
            Maximum number of extra data points before and after the 
            cycle's beginning and end for smoothing. Additional points
            will be discarded. Default is 8.

        Returns
        -------
        None
        """
        self.cycle_id = cycle_id
        
        # Data which will stored and not modified.
        self.cell_data = mother_data.copy()
        self.previous_bud_data = previous_bud_data.copy()
        self.current_bud_data = current_bud_data.copy()
        self.cycle_events = cycle_events.copy()
        self.cycle_end_event = cycle_end_event

        self.validate_input_data(min_extra_data_points, max_extra_data_points)
        
        # Initialize internal data containers for calculated values - these will
        # be populated by analysis methods and typically acccessed via properties.
        self._cycle_data: Optional[pd.DataFrame] = None
        self._abundance_gp: Optional[GaussianProcessRegressor] = None
        self._volume_gp: Optional[GaussianProcessRegressor] = None

    def __bool__(self) -> bool:
        """
        Return True if the cycle data has been merged, False otherwise.

        Returns
        -------
        bool
            True if cycle data is available, False if not.
        """
        return self._cycle_data is not None
    
    def __eq__(self, other) -> bool:
        """
        Check equality based on cycle ID and cycle data values.
        Two CellCycle instances are considered equal if they have the
        same cycle ID and their cycle data DataFrames are equal. Raises
        and error if the other object is not a CellCycle instance.

        Parameters
        ----------
        other : Any
            The object to compare with this CellCycle instance.
        
        Returns
        -------
        bool
            True if the cycle IDs and cycle data are equal, False 
            otherwise.

        Raises
        ------
        TypeError
            If `other` is not an instance of CellCycle.
        """
        if not isinstance(other, CellCycle):
            raise TypeError(
                f"Cannot compare CellCycle with {type(other).__name__}. "
                "Comparison is only supported between CellCycle instances."
            )
        # Compare cycle IDs and cycle data DataFrames.
        return (
            self.cycle_id == other.cycle_id
            and self._cycle_data.equals(other._cycle_data)
        )
    
    def __getitem__(self, key: str) -> np.ndarray:
        """
        Access cycle data columns by name.

        Parameters
        ----------
        key : str
            The name of the column to access in the cycle data.

        Returns
        -------
        np.ndarray
            The values from the specified column in the cycle data.

        Raises
        ------
        KeyError
            If the specified key does not exist in the cycle data.
        """
        return self._get_cycle_data_column_or_raise(
            key, f"Column '{key}' not found in cycle data."
        )
    
    def __hash__(self):
        """
        Return a hash based on the cycle ID.
        """
        return hash(self.cycle_id)

    def __len__(self) -> int:
        """
        Return the number of time points in the merged cycle data or 
        zero if data has not yet been merged.

        Returns
        -------
        int
            The number of time points in the cycle data.
        """
        if self._cycle_data is None:
            return 0
        return len(self._cycle_data)
    
    def __repr__(self) -> str:
        """
        Provide an informative text representation.
        
        This representation shows the cycle ID and indicates which 
        analysis steps have been completed, giving users quick feedback 
        about the current state.
        """
        output = {"CellCycle" : self.cycle_id}
        # Check which analysis steps have been completed and add to output.
        is_merged = self._cycle_data is not None
        output["Merged"] = is_merged
        # Avoid accessing cycle_data properties if data is not merged as that would
        # raise an error.
        if is_merged:
            output["Abundance"] = self._cycle_data_has_column("Abundance")
            output["Smoothed abundance"] = self._cycle_data_has_column(
                "Smoothed abundance"
            )
            output["Production rate"] = self._cycle_data_has_column("Production rate")
            output["Smoothed volume"] = self._cycle_data_has_column("Smoothed volume")
            output["Specific production rate"] = self._cycle_data_has_column(
                "Specific production rate"
            )
        else:
            output["Abundance"] = False
            output["Smoothed abundance"] = False
            output["Production rate"] = False
            output["Smoothed volume"] = False
            output["Specific production rate"] = False
        return str(output)[1:-1]  # Remove the outer braces for cleaner output.
    

    # Properties for accessing computed data. These help to avoid accidentally
    # changing computed values, make code easier to read, and allow for more helpful 
    # error messages.
    @property
    def cycle_data(self) -> pd.DataFrame:
        """DataFrame containing merged cycle data."""
        if self._cycle_data is None:
            raise ValueError(
                "Attempted to access a value in cycle_data but cycle_data is missing. "
                "Call merge_data() first."
            )
        return self._cycle_data
    
    @cycle_data.setter
    def cycle_data(self, value: Any) -> None:
        """Prevent unintended direct modification of cycle_data."""
        raise AttributeError(
            "CellCycle.cycle_data is read-only and cannot be modified directly. "
            "Usually this is only modified by the CellCycle analysis methods."
            "If you really want to change CellCycle stored data manually, modify the "
            "CellCycle._cycle_data attribute directly instead."
        )

    @property 
    def time_id(self) -> np.ndarray:
        """Imaging frame TimeIDs."""
        return self._get_cycle_data_column_or_raise(
            "TimeID", "TimeID not available. Call merge_data() first."
        )
    
    @property 
    def time(self) -> np.ndarray:
        """Time points in minutes from cycle start."""
        return self._get_cycle_data_column_or_raise(
            "Time", "Time not calculated. Call merge_data() first."
        )
    
    @property
    def total_volume(self) -> np.ndarray:
        """Total volume (cell + buds) at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Total volume", "Total volume not calculated. Call merge_data() first."
        )
    @property
    def cell_volume(self) -> np.ndarray:
        """Volume of the mother cell at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Volume", "Cell volume not available. Call merge_data() first."
        )
    
    @property
    def previous_bud_volume(self) -> np.ndarray:
        """Volume of the previous bud at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Previous bud volume", 
            "Previous bud volume not available. Call merge_data() first."
        )
    
    @property
    def current_bud_volume(self) -> np.ndarray:
        """Volume of the current bud at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Current bud volume", 
            "Current bud volume not available. Call merge_data() first."
        )
    
    @property
    def smoothed_volume(self) -> np.ndarray:
        """Gaussian process smoothed volume estimates."""
        return self._get_cycle_data_column_or_raise(
            "Smoothed volume", 
            "Volume not smoothed. Call calculate_smoothed_volume() first."
        )
    
    @property
    def volume_std(self) -> np.ndarray:
        """Standard deviation of the smoothed volume estimates."""
        return self._get_cycle_data_column_or_raise(
            "Volume std", 
            "Volume standard deviation not calculated. "
            "Call calculate_smoothed_volume() first."
        )
    
    @property
    def concentration(self) -> np.ndarray:
        """Concentration of the fluorophore."""
        return self._get_cycle_data_column_or_raise(
            "Concentration", "Concentration data not available. Call merge_data() first."
        )
    
    @property
    def abundance(self) -> np.ndarray:
        """Fluorescent protein abundance at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Abundance", 
            "Abundance not calculated. Call calculate_abundance() first."
        )
    
    @property
    def smoothed_abundance(self) -> np.ndarray:
        """
        Fluorescent protein abundance estimates smoothed with a 
        Gaussian process.
        """
        return self._get_cycle_data_column_or_raise(
            "Smoothed abundance", 
            "Abundance not smoothed. Call calculate_smoothed_abundance() first."
        )
    
    @property
    def abundance_std(self) -> np.ndarray:
        """
        Standard deviation of the smoothed fluorescent protein 
        abundance estimates.
        """
        return self._get_cycle_data_column_or_raise(
            "Abundance std", 
            "Abundance standard deviation not calculated. "
            "Call calculate_smooth_abundance() first."
        )
    
    @property
    def production_rate(self) -> np.ndarray:
        """Fluorescent protein production rate at each time point."""
        return self._get_cycle_data_column_or_raise(
            "Production rate", 
            "Production rate not calculated. Call calculate_production_rate() first."
        )
    
    @property
    def specific_production_rate(self) -> np.ndarray:
        """
        Volume-specific fluorescent protein production rate at each time
        point.
        """
        return self._get_cycle_data_column_or_raise(
            "Specific production rate", 
            "Volume-specific production rate not calculated. "
            "Call calculate_volume_specific_production_rate() first."
        )
    
    @property
    def abundance_gp(self) -> GaussianProcessRegressor:
        """GaussianProcessRegressor for smoothed abundance."""
        if self._abundance_gp is None:
            raise ValueError(
                "Abundance Gaussian process not fitted. "
                "Call calculate_smooth_abundance() first."
            )
        return self._abundance_gp
    
    @property
    def volume_gp(self) -> GaussianProcessRegressor:
        """GaussianProcessRegressor for smoothed volume."""
        if self._volume_gp is None:
            raise ValueError(
                "Volume Gaussian process not fitted. "
                "Call calculate_smoothed_volume() first."
            )
        return self._volume_gp
    
    @property
    def previous_bud_time_id(self) -> int:
        """TimeID of the previous bud event."""
        return self.cycle_events["Bud_0"]
    
    @property
    def current_bud_time_id(self) -> int:
        """TimeID of the current bud event."""
        return self.cycle_events["Bud_1"]
    
    @property
    def previous_cycle_end_time_id(self) -> int:
        """TimeID of the previous cycle end event."""
        return self.cycle_events[f"{self.cycle_end_event}_0"]
    
    @property
    def current_cycle_end_time_id(self) -> int:
        """TimeID of the current cycle end event."""
        return self.cycle_events[f"{self.cycle_end_event}_1"]
    
    @property
    def cycle_duration(self) -> float:
        """Duration of the cell cycle in minutes."""
        time = self.time
        mask = self._mask_time_ids_between(
            self.previous_cycle_end_time_id, 
            self.current_cycle_end_time_id, 
            "both"
        )
        return time[mask].max() - time[mask].min()
    

    # Core analysis methods - these form the main pipeline.
    # Each method stores the results, generally in the _cycle_data
    # DataFrame, and returns self to enable method chaining.
    def merge_cycle_data(
            self,
            image_capture_interval: int,
            max_extra_data_points: int = 8
        ) -> Self:
        """
        Merge cell and bud data into a unified time series.

        This method combines volume data from the mother cell and both 
        buds into a single DataFrame. It handles interpolation of 
        flagged data points, adjusts bud volumes based on cell cycle 
        events, and calculates time values in minutes from TimeIDs.

        Parameters
        ----------
        image_capture_interval : int
            The interval between image captures, in minutes. Used to 
            calculate time values.
        max_extra_data_points : int, optional
            The maximum number of extra data points to include before 
            and after the relevant cell cycle time range.
            Default is 8.

        Returns
        -------
        Self
            The instance of the class with the merged cycle data stored 
            and accessible from the .cycle_data attribute.

        Notes
        -----
        The merged cycle data is clipped to have no more than 
        `max_extra_data_points` before the previous cycle end and
        after the current cycle end. In the case that Bud_0 is None,
        points will be clipped before the current cycle if they are
        missing from either the mother or previous bud data.
        """
        # Prepare data for merging. Delete any datapoints which should be removed,
        # interpolate any resulting missing values, and then remove the redundant 
        # Interpolate column.
        cell_data = self.cell_data.copy()
        cell_data.loc[cell_data["Interpolate"], ["Volume", "Concentration"]] = np.nan
        cell_data.drop(columns=["Interpolate"], inplace=True)

        previous_bud_data = self.previous_bud_data.copy()
        previous_bud_data.loc[previous_bud_data["Interpolate"], "Volume"] = np.nan
        previous_bud_data.drop(columns=["Interpolate"], inplace=True)

        current_bud_data = self.current_bud_data.copy()
        current_bud_data.loc[current_bud_data["Interpolate"], "Volume"] = np.nan
        current_bud_data.drop(columns=["Interpolate"], inplace=True)

        # Combine the data into a single DataFrame.
        merged_data = cell_data.merge(
            previous_bud_data.rename(columns={"Volume": "Previous bud volume"}),
            on="TimeID",
            how="left"
        ).merge(
            current_bud_data.rename(columns={"Volume": "Current bud volume"}),
            on="TimeID",
            how="left"
        )
        # Setting the index like this allows for easy access to values with specific
        # TimeIDs using the .at[] accessor.
        merged_data.set_index(merged_data["TimeID"].values, inplace=True)

        # Ensure that bud volumes are set to 0 up to and including the relevant bud
        # events. Skip this for the previous bud if Bud_0 is None.
        if self.previous_bud_time_id is not None:
            pre_bud_mask = merged_data["TimeID"] <= self.previous_bud_time_id
            merged_data.loc[pre_bud_mask, "Previous bud volume"] = 0.0

        pre_bud_mask = merged_data["TimeID"] <= self.current_bud_time_id
        merged_data.loc[pre_bud_mask, "Current bud volume"] = 0.0

        # If Bud_0 is None, remove any time points from the previous cycle where either
        # previous bud data is missing.
        if self.previous_bud_time_id is None:
            mask = (
                (merged_data["TimeID"] < self.current_bud_time_id)
                & (merged_data["Previous bud volume"].isna())
            )
            merged_data = merged_data.loc[~mask]
        
        # Handle any remaining NaN values, particularly the bud volumes from budding up 
        # to the first point at which they were tracked.
        merged_data.interpolate(method="linear", axis="rows", inplace=True)

        # Ensure that bud volumes are fixed after the relevant cell cycle end point.
        previous_bud_final_volume = merged_data.at[
            self.previous_cycle_end_time_id, "Previous bud volume"
        ]
        post_bud_mask = merged_data["TimeID"] > self.previous_cycle_end_time_id
        merged_data.loc[post_bud_mask, "Previous bud volume"] = previous_bud_final_volume

        current_bud_final_volume = merged_data.at[
            self.current_cycle_end_time_id, "Current bud volume"
        ]
        post_bud_mask = merged_data["TimeID"] > self.current_cycle_end_time_id
        merged_data.loc[post_bud_mask, "Current bud volume"] = current_bud_final_volume

        # Adjust previous bud volumes such that correct volumes are maintained over the
        # current cell cycle.
        merged_data["Previous bud volume"] = (
            merged_data["Previous bud volume"] - previous_bud_final_volume
        )

        # Clip unnecessary data points.
        min_required_time_id = (
            self.previous_cycle_end_time_id - max_extra_data_points
        )
        max_required_time_id = (
            self.current_cycle_end_time_id + max_extra_data_points
        )
        time_id_mask = (
            (merged_data["TimeID"] >= min_required_time_id)
            & (merged_data["TimeID"] <= max_required_time_id)
        )
        merged_data = merged_data.loc[time_id_mask]

        # Finalise and store the merged data frame.
        merged_data["Total volume"] = (
            merged_data["Volume"]
            + merged_data["Previous bud volume"]
            + merged_data["Current bud volume"]
        )
        merged_data["Time"] = (merged_data["TimeID"] - 1) * image_capture_interval
        self._cycle_data = merged_data
        return self
    
    def calculate_abundance(self) -> Self:
        """
        Calculate fluorescent protein abundance by multiplying 
        concentration and total volume.

        Returns
        -------
        Self
            The instance of the class with the calculated abundance 
            stored in the cycle data.
        """
        abundance = self.concentration * self.total_volume
        self._cycle_data["Abundance"] = abundance
        return self
    
    def calculate_smoothed_abundance(
            self, 
            constant_value: float = 1.0,
            constant_value_bounds: Tuple[float, float] = (0.1, 10),
            length_scale: float = 10.0,
            length_scale_bounds: Tuple[float, float] = (1.0, 200.0),
            alpha: float = 1.0,
            alpha_bounds: Tuple[float, float] = (0.1, 1e7),
            noise_level: float = 0.001,
            noise_level_bounds: Tuple[float, float] = (1e-4, 1.0),
            gp_alpha: float = 1e-10,
            n_restarts: int = 1,
            random_seed: int = 42
        ) -> Self:
        """
        Apply Gaussian process smoothing to abundance estimates.

        Parameters
        ----------
        constant_value : float, optional
            Initial value for the constant kernel. Default is 1.0.
        constant_value_bounds : Tuple[float, float], optional
            Bounds for the constant kernel value. Default is (0.1, 10).
        length_scale : float, optional
            Initial length scale for the Rational Quadratic kernel. 
            Default is 10.0.
        length_scale_bounds : Tuple[float, float], optional
            Bounds for the length scale of the Rational Quadratic 
            kernel. Default is (1.0, 200.0).
        alpha : float, optional
            Initial alpha value for the Rational Quadratic kernel, which
            determines the relative weighting of large-scale and 
            small-scale variations. Default is 1.0.
        alpha_bounds : Tuple[float, float], optional
            Bounds for the alpha parameter of the Rational Quadratic 
            kernel. Default is (0.1, 1e7).
        noise_level : float, optional
            Initial noise level for the White kernel. Default is 0.001.
        noise_level_bounds : Tuple[float, float], optional
            Bounds for the noise level of the White kernel. 
            Default is (1e-4, 1.0).
        gp_alpha : float, optional
            Value added to the diagonal of the kernel matrix during 
            fitting to improve numerical stability. Default is 1e-10.
        n_restarts : int, optional
            Number of restarts for the optimizer to find the best kernel
            parameters. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        Self
            The instance of the class with the smoothed abundance
            estimates and their standard deviation stored in the cycle
            data. The GaussianProcessRegressor model is also stored.
        """
        # This method uses a composite Gaussian process kernel to smooth abundance 
        # estimates over time. The kernel consists of a Constant kernel to capture 
        # the overall scale, a Rational Quadratic kernel to model long-term dynamic 
        # trends, and a White kernel to approximate short-term noise. The smoothed 
        # abundance and its standard deviation are stored in the cycle data.
        constant_kernel = ConstantKernel(constant_value, constant_value_bounds)
        rq_kernel = RationalQuadratic(
            length_scale, alpha, length_scale_bounds, alpha_bounds
        )
        white_kernel = WhiteKernel(noise_level, noise_level_bounds)
        gp_kernel = constant_kernel * rq_kernel + white_kernel
        gp_regressor = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=gp_alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
            random_state=random_seed
        )

        gp_time = self.time[:, np.newaxis]
        gp_abundance = self.abundance[:, np.newaxis]
        mean_abundance = self.abundance.mean()
        # Subtract the center and scale data around 0 to produce better fits. Reverse
        # this transformation after fitting.
        gp_fit = gp_regressor.fit(gp_time, (gp_abundance / mean_abundance) - 1)
        smooth_abundance, abundance_std = gp_fit.predict(gp_time, return_std=True)
        smooth_abundance = (smooth_abundance + 1) * mean_abundance
        abundance_std = abundance_std * mean_abundance

        # Store the Gaussian process model and smoothed abundance in the cycle data.
        self._abundance_gp = gp_fit
        self._cycle_data["Smoothed abundance"] = smooth_abundance
        self._cycle_data["Abundance std"] = abundance_std
        return self
    
    def calculate_production_rate(
            self, 
            apply_maturation_correction: bool = False,
            maturation_time: Optional[float] = None
        ) -> Self:
        """
        Calculate the fluorescent protein production rate.

        This method estimates the production rate of a fluorescent 
        protein based on the smoothed abundance data. It also calculates
        the relative production rate by dividing the production rate by
        the mean value. Optionally, it can apply a correction for the 
        fluorophore maturation time.

        Parameters
        ----------
        apply_maturation_correction : bool, optional
            If False, applies a correction for the fluorophore 
            maturation time. Default is False.
        maturation_time : float, optional
            The maturation time of the fluorophore, required if 
            `apply_maturation_correction` is False.

        Returns
        -------
        Self
            The instance of the class with the calculated production 
            rate stored in cycle data.

        Raises
        ------
        ValueError
            If `apply_maturation_correction` is True and 
            `maturation_time` is not provided.
        """
        # Estimate derivatives for calculating fluorophore production rate. 
        # Predict abundance values from the stored GaussianProcessRegressor using
        # a high sample rate to ensure smooth derivatives.
        gp_time = self.time[:, np.newaxis]
        steps_per_timepoint = 10
        steps = (steps_per_timepoint * (len(gp_time) - 1)) + 1
        derivative_times, dt = np.linspace(
            gp_time.min(), gp_time.max(), steps, retstep=True
        )
        smooth_abundance = (
            self.abundance_gp.predict(derivative_times[:, np.newaxis]).squeeze()
        )
        # Compensate for the scaling and centering performed during fitting.
        smooth_abundance = (smooth_abundance + 1) * self.abundance.mean()

        # Calculate the rate of abundance change.
        abundance_1st_derivative = np.gradient(smooth_abundance, dt)
        if apply_maturation_correction:
            if maturation_time is None:
                raise ValueError(
                    "Maturation time must be provided for maturation correction."
                )
            abundance_2nd_derivative = np.gradient(abundance_1st_derivative, dt)
            k_maturation = np.log(2)/maturation_time
            production_rate = (
                abundance_1st_derivative
                + ((1/k_maturation) * abundance_2nd_derivative)
            )
        else:
            production_rate = abundance_1st_derivative

        # Downsample the rate of change to match the original time points and store.
        production_rate = production_rate[::steps_per_timepoint]
        self._cycle_data["Production rate"] = production_rate
        self._cycle_data["Normalised production rate"] = (
            production_rate / production_rate.mean()
        )
        return self
    
    def calculate_smoothed_volume(
            self,
            constant_value: float = 10.0,
            constant_value_bounds: Tuple[float, float] = (1.0, 1000.0),
            length_scale: float = 200.0,
            length_scale_bounds: Tuple[float, float] = (10.0, 1000.0),
            noise_level: float = 1.0,
            noise_level_bounds: Tuple[float, float] = (0.01, 100.0),
            gp_alpha: float = 1e-10,
            n_restarts: int = 1, 
            random_seed: int = 42
        ) -> Self:
        """
        Apply Gaussian process smoothing to volume estimates.

        Parameters
        ----------
        constant_value : float, optional
            Initial value for the constant kernel. Default is 10.0.
        constant_value_bounds : Tuple[float, float], optional
            Bounds for the constant kernel value. 
            Default is (1.0, 1000.0).
        length_scale : float, optional
            Initial length scale for the RBF kernel. Default is 200.0.
        length_scale_bounds : Tuple[float, float], optional
            Bounds for the RBF kernel length scale. 
            Default is (10.0, 1000.0).
        noise_level : float, optional
            Initial noise level for the White kernel. Default is 1.0.
        noise_level_bounds : Tuple[float, float], optional
            Bounds for the White kernel noise level. 
            Default is (0.01, 100.0).
        gp_alpha : float, optional
            Value added to the diagonal of the kernel matrix during 
            fitting to improve numerical stability. Default is 1e-10.
        n_restarts : int, optional
            Number of restarts for the optimizer to find the best kernel
            parameters. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        Self
            The instance of the class with updated smoothed volume data
            stored in the cycle data. The GaussianProcessRegressor model
            is also stored.
        """
        # Perform smoothing using a composite Gaussian process kernel. The Constant
        # kernel captures the overall scale, The RBF kernel captures 
        # the smooth longer-term dynamic trends, while the White kernel approximates 
        # short term noise.
        constant_kernel = ConstantKernel(constant_value, constant_value_bounds)
        rbf_kernel = RBF(length_scale, length_scale_bounds)
        white_kernel = WhiteKernel(noise_level, noise_level_bounds)
        gp_kernel = constant_kernel * rbf_kernel + white_kernel
        gp_regressor = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=gp_alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
            random_state=random_seed
        )

        gp_time = self.time[:, np.newaxis]
        gp_volume = self.total_volume[:, np.newaxis]
        mean_volume = self.total_volume.mean()
        # Subtract center the data around 0 produce better fits. In this case scaling
        # generally doesn't seem to be required and not doing so allows use of more
        # intuitive noise bounds values. Reverse this transformation after fitting.
        gp_fit = gp_regressor.fit(gp_time, gp_volume - mean_volume)
        smooth_volume, volume_std = gp_fit.predict(gp_time, return_std=True)
        smooth_volume += mean_volume

        # Store the Gaussian process model and smoothed volume in the cycle data.
        self._volume_gp = gp_fit
        self._cycle_data["Smoothed volume"] = smooth_volume
        self._cycle_data["Volume std"] = volume_std
        return self
    
    def calculate_volume_specific_production_rate(self) -> Self:
        """
        Calculate volume-specific fluorescent protein prodution rate.

        Returns
        -------
        Self
            The instance of the class with the volume-specific 
            production rate stored in the cycle data.
        """
        specific_production_rate = self.production_rate / self.smoothed_volume
        self._cycle_data["Specific production rate"] = specific_production_rate
        return self
    
    def calculate_all_cycle_values(
            self, 
            image_capture_interval: int,
            max_extra_data_points: int = 8,
            calculate_smoothed_abundance_kwargs: Dict[str, Any] = {},
            calculate_production_rate_kwargs: Dict[str, Any] = {},
            calculate_smoothed_volume_kwargs: Dict[str, Any] = {}
            ) -> Self:
        """
        Convenience method to run the complete analysis pipeline.

        This method performs a series of calculations on cycle data, 
        including merging data, calculating abundance, 
        smoothing abundance, calculating production rates, smoothing 
        volume, and calculating volume-specific production rates. It 
        allows for customization of the parameters used in the smoothing
        and production rate calculations.

        Parameters
        ----------
        image_capture_interval : int
            The interval (in minutes or other time units) at which 
            images were captured.
        max_extra_data_points : int, optional
            The maximum number of extra data points to include. 
            Default is 8.
        calculate_smoothed_abundance_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_smoothed_abundance` method.
            Default is an empty dictionary.
        calculate_production_rate_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_production_rate` method.
            Default is an empty dictionary.
        calculate_smoothed_volume_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_smoothed_volume` method. 
            Default is an empty dictionary.

        Returns
        -------
        Self
            The instance of the class with all cycle values calculated
            and stored as appropriate.
        """
        (
            self
            .merge_cycle_data(image_capture_interval, max_extra_data_points)
            .calculate_abundance()
            .calculate_smoothed_abundance(**calculate_smoothed_abundance_kwargs)
            .calculate_production_rate(**calculate_production_rate_kwargs)
            .calculate_smoothed_volume(**calculate_smoothed_volume_kwargs)
            .calculate_volume_specific_production_rate()
        )
        return self
    
    def align_to_standard_coordinate(self, event_anchors: Dict[str, float]) -> Self:
        """
        Map time points to standardized cell cycle progression 
        coordinates.

        This method maps the time points of a cell cycle to a 
        standardized coordinate system based on the provided 
        `event_anchors`. The mapping assumes that the `event_anchors` 
        are supplied in increasing order, corresponding to the 
        biological ordering of cell cycle events.

        Parameters
        ----------
        event_anchors : Dict[str, float]
            A dictionary where keys are event names (e.g., "Bud_1") and 
            values are the corresponding standardized coordinates. The 
            keys must match the event names in `self.cycle_events`.

        Returns
        -------
        Self
            The instance of the class with the standardized coordinates
            stored in the cycle data.

        Raises
        ------
        ValueError
            If any event key in `event_anchors` is not present in 
            `self.cycle_events`.

        Examples
        --------
        >>> event_anchors = {
        >>>     "Cytokinesis_0": 0.0, "Bud_0": 0.3, "Cytokinesis_1": 1.0
        >>> }
        >>> cycle.align_to_standard_coordinate(event_anchors)
        """
        # This will be inserted into the cycle_data DataFrame so the lengths must match.
        # However, the standard coordinate will probably not cover the full range of
        # time points. Initialise to NaN so those points will be NaN in the finished
        # array.
        standard_coordinates = np.zeros(len(self.time)) * np.nan

        anchor_event_keys = tuple(event_anchors.keys())
        for i, current_event_key in enumerate(anchor_event_keys):
            # Ensure that the required event key is present in cycle_events.
            if current_event_key not in self.cycle_events:
                raise ValueError(
                    f"Cycle {self.cycle_id} missing event '{current_event_key}' for "
                    "coordinate mapping."
                )
            
            # Skip the first event since we interpolate to an event from the previous
            # event which is not possible with the first one.
            if i == 0:
                continue

            # Linearly interpolate between event anchor standard coordinates.
            previous_event_key = anchor_event_keys[i - 1]
            begin_time_id = self.cycle_events[previous_event_key]
            end_time_id = self.cycle_events[current_event_key]
            # Determine the number of points required for interpolation.
            mask = self._mask_time_ids_between(begin_time_id, end_time_id, "both")
            n_points = int(mask.sum())
            # Compute coordinates for the current segment.
            new_coords = np.linspace(
                    event_anchors[previous_event_key], 
                    event_anchors[current_event_key], 
                    n_points
                )
            standard_coordinates[mask] = new_coords

        # Store the values in the cycle data.
        self._cycle_data["Standard coordinate"] = standard_coordinates
        return self
    

    # Plotting methods for visualization and validation. Users should use the .plot()
    # method. The other methods starting with "_" are intended for internal use.
    def plot(
            self, 
            plot_type: str,
            figsize: Optional[Tuple[float, float]] = None,
            add_title: bool = True,
            show_events: bool = True,
            show_events_in_legend: bool = True,
        ) -> Figure:
        """
        Plot cycle data for validation and visualization.

        Parameters
        ----------
        plot_type : str
            The type of plot to generate. Supported values are:
            - "volume": Plots the volume data.
            - "concentration": Plots the concentration data.
            - "abundance": Plots the abundance data.
            - "production rate": Plots the production rate data.
            - "overview": Provides an overview plot of all data.
        figsize : Tuple[float, float], optional
            The size of the figure in inches. If None, uses (10.0, 6.0)
            for all plot types other than "overview" which instead uses
            (20.0, 12.0). Default is None.
        add_title : bool, optional
            Whether to add a title to the plot. If plot_type is
            "overview", adds the title as the figure suptitle instead.
            Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plot. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend. If True
            while the plot_type is "overview", the events will only be
            noted in the legend of the concentration plot.
            Default is True.

        Returns
        -------
        Figure
            A matplotlib Figure object containing the generated plot.

        Raises
        ------
        ValueError
            If `plot_type` is not one of the supported values.
        """
        match plot_type.strip().lower():
            case "volume":
                plot_func = self._plot_volume
                figsize = (10.0, 6.0) if figsize is None else figsize
            case "concentration":
                plot_func = self._plot_concentration
                figsize = (10.0, 6.0) if figsize is None else figsize
            case "abundance":
                plot_func = self._plot_abundance
                figsize = (10.0, 6.0) if figsize is None else figsize
            case "production rate":
                plot_func = self._plot_production_rate
                figsize = (10.0, 6.0) if figsize is None else figsize
            case "overview":
                plot_func = self._plot_overview
                figsize = (20.0, 12.0) if figsize is None else figsize
            case _:
                raise ValueError(
                    f"Unknown plot type '{plot_type}'. "
                    "Valid options are: 'Volume', 'Concentration', 'Abundance', "
                    "'Production Rate', 'Overview'."
                )
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        plot_func(ax, add_title, show_events, show_events_in_legend)
        return fig
    
    def _plot_volume(
            self, 
            ax: Axes, 
            add_title: bool = True, 
            show_events: bool = True,
            show_events_in_legend = True
        ) -> None:
        """
        Plot volume data for validation as well as smoothed volumes if 
        available.

        Parameters
        ----------
        ax : Axes
            The matplotlib Axes object to plot on.
        add_title : bool, optional
            Whether to add a title to the plot. Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plot. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend.
            Default is True.

        Returns
        -------
        None
        """
        ax.plot(self.time, self.total_volume, marker="x", label="Total volume")
        ax.plot(self.time, self.cycle_data["Volume"], marker="x", label="Cell volume")

        # Only plot the bud volumes while the buds are present.
        previous_bud_mask = self._mask_time_ids_between(
            self.previous_bud_time_id, self.previous_cycle_end_time_id, "both"
        )
        # Compensate for previous bud volume adjustment to give a more intuitive plot.
        previous_bud_final_volume = self.previous_bud_data.loc[
            self.previous_bud_data["TimeID"] == self.previous_cycle_end_time_id,
            "Volume"
        ].values[0]
        ax.plot(
            self.time[previous_bud_mask], 
            self.previous_bud_volume[previous_bud_mask] + previous_bud_final_volume,
            marker="x",
            label="Previous bud volume"
        )

        current_bud_mask = self._mask_time_ids_between(
            self.current_bud_time_id, self.current_cycle_end_time_id, "both"
        )
        ax.plot(
            self.time[current_bud_mask], 
            self.current_bud_volume[current_bud_mask], 
            marker="x", 
            label="Current bud volume"
        )

        # Plot smoothed volume estimates if they are available. Don't raise an error if
        # they are not, because plotting the unsmoothed volumes alone may still be 
        # useful.
        if self._cycle_data_has_column("Smoothed volume"):
            ax.plot(
                self.time, 
                self.smoothed_volume,
                color="black",
                linestyle="-",
                label="Smoothed volume"
            )
        if self._cycle_data_has_column("Volume std"):
            ax.fill_between(
                self.time,
                self.smoothed_volume - self.volume_std,
                self.smoothed_volume + self.volume_std,
                color="black", 
                alpha=0.2, 
                label="Volume St.Dev"
            )
        
        if show_events:
            self._plot_cycle_events(ax)
            legend_items = ax.get_legend_handles_labels()
            if show_events_in_legend:
                pass
                ax.legend(*_deduplicate_legend_items(legend_items))
            else:
                ax.legend(legend_items[0][:7], legend_items[1][:7])
        else:
            ax.legend()
        
        ax.set_xlabel("Time after imaging start (min)")
        ax.set_ylabel("Volume (fL)")
        if add_title:
            ax.set_title(f"Cell Cycle {self.cycle_id} - Volume")
            
    def _plot_concentration(
            self, 
            ax: Axes, 
            add_title: bool = True, 
            show_events: bool = True,
            show_events_in_legend: bool = True
        ) -> None:
        """
        Plot concentration data for validation.
        
        Parameters
        ----------
        ax : Axes
            The matplotlib Axes object to plot on.
        add_title : bool, optional
            Whether to add a title to the plot. Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plot. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend.
            Default is True.

        Returns
        -------
        None
        """
        ax.plot(self.time, self.concentration, marker="x", label="Concentration")
        
        ax.set_xlabel("Time after imaging start (min)")
        ax.set_ylabel("Concentration (a.u.)")

        if show_events:
            self._plot_cycle_events(ax)
            legend_items = ax.get_legend_handles_labels()
            if show_events_in_legend:
                ax.legend(*_deduplicate_legend_items(legend_items))
            else:
                ax.legend(legend_items[0][:1], legend_items[1][:1])
        else:
            ax.legend()

        if add_title:
            ax.set_title(f"Cell Cycle {self.cycle_id} - Concentration")
    
    def _plot_abundance(
            self, 
            ax: Axes, 
            add_title: bool = True,
            show_events: bool = True,
            show_events_in_legend: bool = True
        ) -> None:
        """
        Plot abundance data for validation and smoothed abundance
        if available.

        Parameters
        ----------
        ax : Axes
            The matplotlib Axes object to plot on.
        add_title : bool, optional
            Whether to add a title to the plot. Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plot. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend.
            Default is True.

        Returns
        -------
        None
        """
        ax.plot(self.time, self.abundance, marker="x", label="Abundance")

        # Plot smoothed abundance estimates if they are available. Don't raise an error 
        # if they are not because plotting the unsmoothed abundances alone may still be 
        # useful.
        if self._cycle_data_has_column("Smoothed abundance"):
            ax.plot(
                self.time, 
                self.smoothed_abundance,
                color="black",
                linestyle="-",
                label="Smoothed abundance"
            )

        if self._cycle_data_has_column("Abundance std"):
            ax.fill_between(
                self.time,
                self.smoothed_abundance - self.abundance_std,
                self.smoothed_abundance + self.abundance_std,
                color="black", 
                alpha=0.2, 
                label="Abundance St.Dev"
            )

        if show_events:
            self._plot_cycle_events(ax)
            legend_items = ax.get_legend_handles_labels()
            if show_events_in_legend:
                ax.legend(*_deduplicate_legend_items(legend_items))
            else:
                ax.legend(legend_items[0][:3], legend_items[1][:3])
        else:
            ax.legend()
        
        ax.set_xlabel("Time after imaging start (min)")
        ax.set_ylabel("Abundance (a.u.)")
        if add_title:
            ax.set_title(f"Cell Cycle {self.cycle_id} - Abundance")

    def _plot_production_rate(
            self, 
            ax: Axes, 
            add_title: bool = True,
            show_events: bool = True,
            show_events_in_legend: bool = True
        ) -> None:
        """
        Plot production rate data for validation.

        Parameters
        ----------
        ax : Axes
            The matplotlib Axes object to plot on.
        add_title : bool, optional
            Whether to add a title to the plot. Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plot. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend.
            Default is True.

        Returns
        -------
        None
        
        """
        # Prefer to put the specific production rate on the primary y-axis, if available.
        if self._cycle_data_has_column("Specific production rate"):
            ax.plot(
                self.time, 
                self.specific_production_rate,
                marker="x", 
                label="Specific production rate"
            )

            ax2 = ax.twinx()
            ax2.plot(
                self.time,
                self.production_rate,
                marker="x",
                color="C1",
                label="Production rate",
            )

            if show_events:
                self._plot_cycle_events(ax)
            
            # Combine legends from both axes into one.
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            # Make sure the production rate entries come before any cell cycle event 
            # ones.
            merged_handles = [handles.pop(0), *handles2]
            merged_labels = [labels.pop(0), *labels2]
            if show_events and show_events_in_legend:
                event_handles, event_labels = _deduplicate_legend_items(
                    (handles, labels)
                )
                merged_handles.extend(event_handles)
                merged_labels.extend(event_labels)
            # Put legend on ax2 otherwise it might show up behind the ax2 plotted data.
            ax2.legend(merged_handles, merged_labels, loc="upper right")
            
            ax.set_ylabel("Volume specific production rate\n(a.u. min⁻¹ fL⁻¹)")
            ax2.set_ylabel("Production rate (a.u. min⁻¹)")

        else:
            # If specific production rate is not available, plot the regular abundance
            # rate on the primary y-axis instead.
            ax.plot(self.time, self.production_rate, marker="x", label="Production rate")
            ax.set_ylabel("Production rate (a.u. min⁻¹)")

            if show_events:
                self._plot_cycle_events(ax)
                legend_items = ax.get_legend_handles_labels()
                if show_events_in_legend:
                    ax.legend(_deduplicate_legend_items(legend_items))
                else:
                    ax.legend(legend_items[0][:1], legend_items[1][:1])
            else:
                ax.legend()

        ax.set_xlabel("Time after imaging start (min)")
        if add_title:
            ax.set_title(f"Cell Cycle {self.cycle_id} - Production Rate")

    def _plot_overview(
        self, 
        ax: Axes,
        add_title: bool = True,
        show_events: bool = True,
        show_events_in_legend: bool = True
        ) -> Figure:
        """
        Plot an overview of all key data for this cell cycle.

        Parameters
        ----------
        ax : Axes
            The matplotlib Axes object to plot on. Removes this axis,
            and instead creates a 2x2 grid of subplots on the associated
            Figure.
        add_title : bool, optional
            Whether to add a suptitle to the figure. Default is True.
        show_events : bool, optional
            Whether to display cell cycle events on the plots. 
            Default is True.
        show_events_in_legend : bool, optional
            Whether to include cell cycle events in the legend. If True,
            only the concentration plot will show the events in the
            legend.
            Default is True.

        Returns
        -------
        None
        """
        # Create a 2x2 grid of subplots for the overview after removing the old axes.
        fig = ax.figure
        ax.remove()
        axs = fig.subplots(2, 2)
        
        # If plotting cycle events, put them on all the axes. If showing
        # events in the legend, only show on the Concentration plot since there
        # is the least going on.
        self._plot_volume(axs[0, 0], False, show_events, False)
        self._plot_concentration(axs[0, 1], False, show_events, show_events_in_legend)
        self._plot_abundance(axs[1, 0], False, show_events, False)
        self._plot_production_rate(axs[1, 1], False, show_events, False)
        if add_title:
            fig.suptitle(f"Cell Cycle {self.cycle_id} Overview")

    def _plot_cycle_events(self, ax: Axes) -> None:
        """Add vertical lines for cycle events to the plot."""
        styles = {
            f"{self.cycle_end_event}" : ("black", "-"),
            "Bud" : ("black", "--")
        }
        for i, (event, time_id) in enumerate(self.cycle_events.items()):
            # Skip the event if it's outside of the range in cycle_data
            if time_id < self.time_id.min() or time_id > self.time_id.max():
                continue

            time = self.cycle_data.at[time_id, "Time"]
            # Cycle end event and Bud will have these appended, remove them.
            if event.endswith(("_0", "_1")):
                event_name = event[:-2]
            else:
                event_name = event
            # Add 4 here to avoid color clash on the volume plot which uses 4 colors
            # already.
            color, linestyle = styles.get(event_name, (f"C{i+4}", ":"))
            ax.axvline(time, color=color, linestyle=linestyle, label=event_name)


    # Methods for validating the input data.
    def validate_input_data(
            self, 
            min_extra_data_points: int = 3,
            max_extra_data_points: int = 8
        ) -> None:
        """
        Validate the input data for structure, consistency, and 
        completeness.

        This method performs a series of checks to ensure that the input
        data frames `cell_data`, `previous_bud_data`, and 
        `current_bud_data` are properly structured, contain the required
        columns, and have consistent and complete data. It also checks
        the supplied `cycle_events`. Errors and warning messages are
        raised as appropriate to inform the user of any issues.

        Parameters
        ----------
        min_extra_data_points : int, optional
            The minimum number of additional data points required before
            and after the cycle end events for smoothing purposes. 
            Default is 3.
        max_extra_data_points : int, optional
            The maximum number of additional data points recommended 
            before or after the cycle end events for smoothing purposes.
            Default is 8.
        """
        self._validate_input_data_frames()
        self._validate_cycle_events()
        self._validate_input_data_frame_time_ids()
        self._validate_sufficient_extra_data_points(
            min_extra_data_points, max_extra_data_points
        )
        
    def _validate_input_data_frames(self) -> None:
        """
        Validate that the input data frames have the required columns.
        Warn if there is any missing data in columns other than TimeID.

        Raises
        ------
        ValueError
            If any of the input `cell_data`, `previous_bud_data`, or
            `current_bud_data` DataFrames are empty, or if they are 
            missing any of the required columns.

        Warns
        -----
        MissingDataWarning
            If any data frame contains missing values in columns other 
            than `TimeID`.
        """
        input_dfs = {
            "cell_data": self.cell_data,
            "previous_bud_data": self.previous_bud_data,
            "current_bud_data": self.current_bud_data
        }
        for name, df in input_dfs.items():
            if df.empty:
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame {name} is empty."
                )
        
        required_cell_cols = {"TimeID", "Volume", "Concentration", "Interpolate"}
        required_bud_cols = {"TimeID", "Volume", "Interpolate"}
        
        if not required_cell_cols.issubset(self.cell_data.columns):
            missing = required_cell_cols - set(self.cell_data.columns)
            raise ValueError(
                f"Cycle {self.cycle_id} data frame cell_data missing required columns: "
                f"{missing}"
            )
        
        for name in ["previous_bud_data", "current_bud_data"]:
            df = input_dfs[name]
            if not required_bud_cols.issubset(df.columns):
                missing = required_bud_cols - set(df.columns)
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} missing columns: {missing}"
                )
        
        # Warn if there are any missing values in columns other than TimeID.
        for name, df in input_dfs.items():
            temp = df.drop(columns=["TimeID"])
            if temp.isna().any().any():
                warn(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} contains missing values in columns other than TimeID.",
                    MissingDataWarning
                )
            
    def _validate_cycle_events(self) -> None:
        """
        Validate the cycle events for consistency and correctness.

        This method checks that the cycle events are correctly ordered,
        that they reference valid TimeIDs, and that all required events
        are present. The only exception is that "Bud_0" may be None to
        accomodate situations where not all previous bud data is
        available. It raises errors if any issues are found.

        Raises
        ------
        ValueError
            If any of the cycle events are missing, incorrectly ordered,
            or reference TimeIDs which are not present in the input
            `cell_data`.
        """
        # Validate that the essential event keys are present in cycle_events and that
        # the corresponding TimeIDs are not overlapping and in the correct order.
        event_keys = [
            "Bud_0", f"{self.cycle_end_event}_0", "Bud_1", f"{self.cycle_end_event}_1"
        ]
        for i, key in enumerate(event_keys):
            if key not in self.cycle_events:
                raise ValueError(
                    f"Cycle {self.cycle_id} missing essential cycle event: '{key}'"
                )
            # Check that TimeIDs are in order. Allow for "Bud_0" to be None.
            if key == "Bud_0" and self.cycle_events["Bud_0"] is None:
                continue
            elif event_keys[i - 1] == "Bud_0" and self.cycle_events["Bud_0"] is None:
                # Can't compare Bud_0 to the previous event, so skip.
                continue
            elif i and self.cycle_events[key] <= self.cycle_events[event_keys[i - 1]]:
                raise ValueError(
                    f"Cycle {self.cycle_id} event '{key}' has TimeID "
                    f"{self.cycle_events[key]} which is less than or equal to the "
                    f"preceding essential event '{event_keys[i - 1]}' which has TimeID "
                    f"{self.cycle_events[event_keys[i - 1]]}."
                )
        
        # Validate that all cycle events reference TimeIDs which have corresponding 
        # data points.
        all_time_ids = set(self.cell_data["TimeID"])
        for key, value in self.cycle_events.items():
            if key == "Bud_0" and value is None:
                continue
            elif value not in all_time_ids:
                raise ValueError(
                    f"Cycle {self.cycle_id} event '{key}' references TimeID: {value} "
                    f"which is not present in cell_data."
                )
    
    def _validate_input_data_frame_time_ids(self) -> None:
        """
        Validate the TimeID values in the input data frames for
        consistency and correctness.

        Raises
        ------
        ValueError
            If any of the following conditions are met:
            - `TimeID` values in any data frame are missing, duplicated,
              skipped, or not in increasing order.
            - `previous_bud_data` or `current_bud_data` contain `TimeID`
              values occuring before the relevant bud events in 
              `cycle_events`.
        """
        input_dfs = {
            "cell_data": self.cell_data,
            "previous_bud_data": self.previous_bud_data,
            "current_bud_data": self.current_bud_data
        }

        # Validate that there are no missing, skipped, duplicated or incorrectly ordered 
        # TimeID values.
        for name, df in input_dfs.items():
            if df["TimeID"].isna().any():
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} contains NaN TimeIDs."
                )
            
            expected_time_ids = set(range(df["TimeID"].min(), df["TimeID"].max() + 1))
            if set(df["TimeID"]) != expected_time_ids:
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} has missing or incorrect TimeIDs."
                )
            
            if df["TimeID"].duplicated().any():
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} contains duplicate TimeIDs."
                )
            
            if df["TimeID"].is_monotonic_increasing is False:
                raise ValueError(
                    f"Cycle {self.cycle_id} data frame "
                    f"{name} TimeIDs are not in increasing order."
                )
            
        # Validate that bud data TimeIDs are within the required ranges.
        min_time_id = self.previous_bud_data["TimeID"].min()
        max_time_id = self.previous_bud_data["TimeID"].max()
        # Allow for "Bud_0" to be None.
        if self.cycle_events["Bud_0"] is None:
            pass
        elif min_time_id < self.cycle_events["Bud_0"]:
            raise ValueError(
                f"Cycle {self.cycle_id} previous_bud_data TimeIDs begin at "
                f"{min_time_id}, which is before the bud event in the previous cycle: "
                f"(Bud_0: {self.cycle_events['Bud_0']})."
            )
        if max_time_id < self.cycle_events[f"{self.cycle_end_event}_0"]:
            raise ValueError(
                f"Cycle {self.cycle_id} previous_bud_data TimeIDs end at "
                f"{max_time_id}, which is before the cycle end event of "
                f"the previous cycle: ({self.cycle_end_event}_0: "
                f"{self.cycle_events[f'{self.cycle_end_event}_0']})."
            )
        
        min_time_id = self.current_bud_data["TimeID"].min()
        max_time_id = self.current_bud_data["TimeID"].max()
        if min_time_id < self.cycle_events["Bud_1"]:
            raise ValueError(
                f"Cycle {self.cycle_id} current_bud_data TimeIDs begin at "
                f"{min_time_id}, which is before the bud event in this cycle: "
                f"(Bud_1: {self.cycle_events['Bud_1']})."
            )
        if max_time_id < self.cycle_events[f"{self.cycle_end_event}_1"]:
            raise ValueError(
                f"Cycle {self.cycle_id} current_bud_data TimeIDs end at "
                f"{max_time_id}, which is before the cycle end event of "
                f"this cycle: ({self.cycle_end_event}_1: "
                f"{self.cycle_events[f'{self.cycle_end_event}_1']})."
            )
        
    def _validate_sufficient_extra_data_points(
            self,
            min_extra_data_points: int = 3,
            max_extra_data_points: int = 8
        ) -> None:
        """
        Validate that the cycle data has sufficient extra data points
        before and after the cycle end events for smoothing purposes. 
        Also checks the previous bud data in the event that "Bud_0" is
        None.
        
        Parameters
        ----------
        min_extra_data_points : int, optional
            The minimum number of additional data points required before
            and after the cycle end events for smoothing purposes. 
            Default is 3.
        max_extra_data_points : int, optional
            The maximum number of additional data points recommended 
            before or after the cycle end events for smoothing purposes.
            Default is 8.

        Raises
        ------
        ValueError
            Extra data points before or after cycle end events is fewer
            than the minimum.

        Warns
        -----
        InsufficientDataWarning
            If the number of additional data points before or after 
            cycle end events is less than the recommended maximum but 
            greater than or equal to the minimum.
        """
        previous_cycle_end_time_id = self.cycle_events[f"{self.cycle_end_event}_0"]
        extra_previous_data_points = (
            previous_cycle_end_time_id - self.cell_data["TimeID"].min()
        )
        if extra_previous_data_points < min_extra_data_points:
            raise ValueError(
                f"Cycle {self.cycle_id} cell_data only has {extra_previous_data_points} "
                f"data points before the previous cycle end event "
                f"which is less than the minimum of {min_extra_data_points}."
            )
        if extra_previous_data_points < max_extra_data_points:
            warn(
                f"Cycle {self.cycle_id} cell_data only has {extra_previous_data_points} "
                f"data points before the previous cycle end event "
                f"which is less than the maximum of {max_extra_data_points}.",
                InsufficientDataWarning
            )
        # If "Bud_0" is None, need to verify that there are sufficient extra
        # data points in the previous bud data because we can't interpolate back
        # to the previous bud event.
        if self.cycle_events["Bud_0"] is None:
            extra_previous_bud_data_points = (
                previous_cycle_end_time_id - self.previous_bud_data["TimeID"].min()
            )
            if extra_previous_bud_data_points < min_extra_data_points:
                raise ValueError(
                    f"Cycle {self.cycle_id} Bud_0 is None and previous_bud_data "
                    f"only has {extra_previous_bud_data_points} data points before the "
                    f"previous cycle end event which is less than the minimum of "
                    f"{min_extra_data_points}."
                )
            if extra_previous_bud_data_points < max_extra_data_points:
                warn(
                    f"Cycle {self.cycle_id} Bud_0 is None and previous_bud_data "
                    f"only has {extra_previous_data_points} data points before the " 
                    f"previous cycle end event which is less than the maximum of "
                    f"{max_extra_data_points}.",
                    InsufficientDataWarning
                )

        current_cycle_end_time_id = self.cycle_events[f"{self.cycle_end_event}_1"]
        extra_current_data_points = (
            self.cell_data["TimeID"].max() - current_cycle_end_time_id
        )
        if extra_current_data_points < min_extra_data_points:
            raise ValueError(
                f"Cycle {self.cycle_id} cell_data only has {extra_current_data_points} "
                f"data points after the current cycle end event "
                f"which is less than the minimum of {min_extra_data_points}."
            )
        if extra_current_data_points < max_extra_data_points:
            warn(
                f"Cycle {self.cycle_id} cell_data only has {extra_current_data_points} "
                f"data points after the current cycle end event "
                f"which is less than the maximum of {max_extra_data_points}.",
                InsufficientDataWarning
            )
        
    # Additional methods.
    def get_time_ids_for_events(self, event_names: Sequence[str]) -> np.ndarray[int]:
        """
        Retrieve TimeIDs for specified cycle events.

        Parameters
        ----------
        event_names : Sequence[str]
            A sequence of event names for which the corresponding 
            TimeIDs are to be retrieved.

        Returns
        -------
        np.ndarray[int]
            An array of integers representing the TimeIDs corresponding 
            to the specified event names.

        Raises
        ------
        ValueError
            If any of the specified event names are not found in the 
            `cycle_events` attribute of the object.
        """
        time_ids = []
        for event_name in event_names:
            if event_name not in self.cycle_events:
                raise ValueError(
                    f"Cycle {self.cycle_id} missing event '{event_name}'"
                )
            time_ids.append(self.cycle_events[event_name])
        return np.array(time_ids, dtype=int)

    def extract_aligned_cycle_data(self) -> pd.DataFrame:
        """
        This method retrieves the aligned cycle data from the cycle 
        dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the aligned cycle data.

        Raises
        ------
        ValueError
            If the "Standard coordinate" column is not present in the 
            cycle data.
        """
        if not self._cycle_data_has_column("Standard coordinate"):
            raise ValueError(
                "Cycle {self.cycle_id} does not have aligned cycle data. "
                "Call align_to_standard_coordinate() first."
            )
        mask = self._cycle_data["Standard coordinate"].notna()
        return self.cycle_data[mask].copy()
    
    def _cycle_data_has_column(self, column_name: str) -> bool:
        """Check if cycle data contains a specific column."""
        return column_name in self.cycle_data.columns

    def _get_cycle_data_column_or_raise(
            self, column_name: str, error_message: str
        ) -> np.ndarray:
        """
        Get a column from cycle data or raise an error if it doesn't 
        exist.
        """
        if not self._cycle_data_has_column(column_name):
            raise ValueError(error_message)
        return self.cycle_data[column_name].values

    def _mask_time_ids_between(
            self, begin: int, end: int, include: str
        ) -> np.ndarray[bool]:
        """Create a boolean mask for TimeIDs between two values."""
        match include:
            case "begin":
                mask = (self.time_id >= begin) & (self.time_id < end)
            case "end":
                mask = (self.time_id > begin) & (self.time_id <= end)
            case "both":
                mask = (self.time_id >= begin) & (self.time_id <= end)
            case "neither":
                mask = (self.time_id > begin) & (self.time_id < end)
            case _:
                raise ValueError(
                    f"Invalid include value: {include}. "
                    "Must be one of 'begin', 'end', 'both', or 'neither'."
                )
        return mask
    
    def _drop_column(self, column: str) -> None:
        """Remove the named column from _cycle_data if it exists."""
        if self._cycle_data_has_column(column):
            self._cycle_data.drop(columns=(column), inplace=True)


class FluorescentProteinProductionRateExperiment:
    """
    Represents a complete experiment with multiple cell cycles.
    
    This class coordinates analysis across multiple cell cycles, 
    providing methods for batch processing and experiment-wide analysis 
    such as Gaussian process fitting across all cycles.

    Notes
    -----
    The main analysis methods for the CellCycle instances can be run for
    all cycles in the experiment using the methods of the same names
    provided by this class. Once a full analysis pipeline has been run
    up to the point of having calculated the volume specific production
    rates, use the `align_to_standard_coordinate()` method to align
    the cycles to a standard coordinate system. The 
    `extract_aligned_cycle_data()` method can then be used to extract
    the aligned cycle data to fit a Gaussian process using 
    `fit_specific_production_rate_gp()`.
    """
    def __init__(
            self, 
            experiment_id: str,
            image_capture_interval: int,
            cycle_end_event: str,
            min_extra_data_points: int = 3,
            max_extra_data_points: int = 8
        ) -> None:
        """
        Initialize an experiment container.
        
        Parameters
        ----------
        experiment_id : str
            A unique identifier for the experiment.
        image_capture_interval : int
            The time interval (in minutes or other units) between 
            consecutive image captures.
        cycle_end_event : str
            The name of the event that marks the end of a cell cycle.
        min_extra_data_points : int, optional
            The minimum number of extra data points before and after
            each cell cycle. Default is 3.
        max_extra_data_points : int, optional
            The maximum number of extra data points which before and
            after each cell cycle. Default is 8.

        Returns
        -------
        None
        """
        self.experiment_id = experiment_id
        self.image_capture_interval = image_capture_interval
        self.cycle_end_event = cycle_end_event
        self.min_extra_data_points = min_extra_data_points
        self.max_extra_data_points = max_extra_data_points
        
        # Container for all cell cycles in this experiment
        self._cell_cycles: Dict[str, CellCycle] = {}

        # Cell cycle event positions on a standard coordinate system used for aligning
        # cell cycles.
        self._standard_coordinate_anchors: Optional[Dict[str, float]] = None
        
        # Experiment-wide analysis results
        self._aligned_cycle_data: Optional[pd.DataFrame] = None
        self._production_rate_gp: Optional[GaussianProcessRegressor] = None
        self._normalised_production_rate_gp: Optional[GaussianProcessRegressor] = None
        self._specific_production_rate_gp: Optional[GaussianProcessRegressor] = None

    def __bool__(self) -> bool:
        """Check if the experiment has any cell cycles."""
        return bool(self.cell_cycles)
    
    def __contains__(self, cycle_id: str) -> bool:
        """
        Check if a cell cycle with the given ID exists in the experiment.

        Parameters
        ----------
        cycle_id : str
            The unique identifier for the cell cycle to check.

        Returns
        -------
        bool
            True if the cell cycle exists, otherwise False.
        """
        return cycle_id in self.cell_cycles
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two experiments are equal based on their IDs and cycle
        data for each of their cell cycles.

        Parameters
        ----------
        other : object
            The object to compare against. Must be an instance of 
            FluorescentProteinProductionRateExperiment.

        Returns
        -------
        bool
            True if the experiments have the same ID and all cell cycles
            are equal, otherwise False.
        
        Raises
        ------
        TypeError
            If the `other` object is not an instance of 
            FluorescentProteinProductionRateExperiment.
        """
        if not isinstance(other, FluorescentProteinProductionRateExperiment):
            raise TypeError(
                f"Cannot compare FluorescentProteinProductionRateExperiment with "
                f"{type(other).__name__}."
            )
        if self.experiment_id != other.experiment_id:
            return False
        if len(self.cell_cycles) != len(other.cell_cycles):
            return False
        for self_cycle, other_cycle in zip(
                self.cell_cycles.values(), other.cell_cycles.values()
            ):
            if self_cycle != other_cycle:
                return False
        return True
    
    def __hash__(self) -> int:
        """Return a hash based on the experiment ID."""
        return hash(self.experiment_id)
    
    def __getitem__(self, cycle_id: str) -> CellCycle:
        """
        Retrieve a cell cycle by its ID.

        Parameters
        ----------
        cycle_id : str
            The unique identifier for the cell cycle to retrieve.

        Returns
        -------
        CellCycle
            The CellCycle object corresponding to the given cycle_id.
        """
        return self.get_cell_cycle(cycle_id)
    
    def __iter__(self) -> Iterator[CellCycle]:
        """
        Iterate over the cell cycles in the experiment.

        Returns
        -------
        Iterator[CellCycle]
            An iterator over the CellCycle objects in the experiment.
        """
        return iter(self.cell_cycles.values())
    
    def __len__(self) -> int:
        """Return the number of cell cycles in the experiment."""
        return len(self.cell_cycles)
    
    def __repr__(self) -> str:
        """Provide informative text representation."""
        n_cycles = len(self.cell_cycles)
        return (
            f"FluorescentProteinExperiment('{self.experiment_id}', "
            f"{n_cycles} cycles)"
            )


    # Properties for providing access to experiment-wide data and useful error messages.
    @property
    def cell_cycles(self) -> Dict[str, CellCycle]:
        """All cell cycles in the experiment."""
        if self._cell_cycles is None:
            raise ValueError(
                "No cell cycles in this experiment. Cycles can be added "
                "using add_cell_cycle()."
            )
        return self._cell_cycles

    @property
    def standard_coordinate_anchors(self) -> Dict[str, float]:
        """The event anchors for standard coordinate mapping."""
        if not self._standard_coordinate_anchors:
            raise ValueError(
                "No standard coordinate anchors available. Use "
                "calculate_standard_coordinate_anchors() to calculate them."
            )
        return self._standard_coordinate_anchors
    
    @property
    def aligned_cycle_data(self) -> pd.DataFrame:
        """Aligned cycle data for the experiment."""
        if self._aligned_cycle_data is None:
            raise ValueError(
                "Aligned cycle data not available. "
                "Call extract_aligned_cycle_data() first."
            )
        return self._aligned_cycle_data
    
    @property
    def production_rate_gp(self) -> GaussianProcessRegressor:
        """
        Gaussian process model for production rate. Note that the GP
        is fitted to mean-scaled and centered values.
        """
        if self._production_rate_gp is None:
            raise ValueError(
                "Production rate Gaussian process not fitted. "
                "Call fit_production_rate_gp(rate_type='basic') first."
            )
        return self._production_rate_gp
    
    @property
    def normalised_production_rate_gp(self) -> GaussianProcessRegressor:
        """
        Gaussian process model for the per-cycle normalised production 
        rate. Note that the GP is fitted to mean-scaled and centered 
        values.
        """
        if self._normalised_production_rate_gp is None:
            raise ValueError(
                "Normalised production rate Gaussian process not fitted. "
                "Call fit_production_rate_gp(rate_type='normalised') first."
            )
        return self._normalised_production_rate_gp
    
    @property
    def specific_production_rate_gp(self) -> GaussianProcessRegressor:
        """
        Gaussian process model for volume specific production rate.
        Note that the GP is fitted to mean-scaled and centered 
        values.
        """
        if self._specific_production_rate_gp is None:
            raise ValueError(
                "Volume specific production rate Gaussian process not fitted. "
                "Call fit_specific_production_rate_gp(rate_type='specific') first."
            )
        return self._specific_production_rate_gp
    

    def add_cell_cycle(
            self, 
            cycle_id: str,
            mother_data: pd.DataFrame,
            previous_bud_data: pd.DataFrame,
            current_bud_data: pd.DataFrame,
            cycle_events: Dict[str, int]
        ) -> CellCycle:
        """
        Add a new cell cycle to the experiment.

        Parameters
        ----------
        cycle_id : str
            Unique identifier for the cell cycle.
        mother_data : pd.DataFrame
            DataFrame containing data for the mother cell. Requires at
            least integer TimeID, float Volume, float Concentration and
            boolean Interpolate columns.
        previous_bud_data : pd.DataFrame
            DataFrame containing data for the previous bud. Requires at
            least integer TimeID, float Volume and boolean Interpolate
            columns.
        current_bud_data : pd.DataFrame
            DataFrame containing data for the current bud. Requires at
            least integer TimeID, float Volume and boolean Interpolate
            columns.
        cycle_events : Dict[str, int]
            A dictionary mapping event names to their corresponding time
            points. Requires at least Bud_0 and Bud_1 key value pairs as
            well as pairs for the cycle end events.

        Returns
        -------
        CellCycle
            The newly created CellCycle object.

        Raises
        ------
        ValueError
            If a cell cycle with the given `cycle_id` already exists in 
            the experiment.

        """
        if cycle_id in self.cell_cycles:
            raise ValueError(f"Cell cycle '{cycle_id}' already exists in experiment")
        
        try:
            # Create and add the CellCycle object.
            cell_cycle = CellCycle(
                cycle_id=cycle_id,
                mother_data=mother_data,
                previous_bud_data=previous_bud_data,
                current_bud_data=current_bud_data,
                cycle_events=cycle_events,
                cycle_end_event=self.cycle_end_event,
                min_extra_data_points=self.min_extra_data_points,
                max_extra_data_points=self.max_extra_data_points
            )
        except Exception as e:
            e.add_note(
                f"This error occurred while creating CellCycle with "
                f"cycle_id: {cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        
        self._cell_cycles[cycle_id] = cell_cycle
        return cell_cycle
    
    def get_cell_cycle(self, cycle_id: str) -> CellCycle:
        """
        Retrieve a specific cell cycle by ID.

        Raises
        ------
        KeyError
            If no cell cycle with the given `cycle_id` exists in the 
            experiment.
        """
        if cycle_id not in self.cell_cycles:
            available = list(self.cell_cycles.keys())
            raise KeyError(
                f"Cell cycle '{cycle_id}' not found. Available: {available}"
            )
        return self.cell_cycles[cycle_id]

    # Methods which perform the same operation across all CellCycle instances in
    # the experiment.
    def merge_cycle_data(self) -> Self:
        """
        Merge data for all cell cycles in the experiment individually.

        Returns
        -------
        Self
        """
        try:
            for cycle in self:
                cycle.merge_cycle_data(
                    image_capture_interval=self.image_capture_interval,
                    max_extra_data_points=self.max_extra_data_points
                )
        except Exception as e:
            e.add_note(
                "This error occurred while running merge_cycle_data() for cycle "
                f"{cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_abundance(self) -> Self:
        """
        Calculate abundance for all cell cycles in the experiment 
        individually.

        Returns
        -------
        Self
        """
        try:
            for cycle in self:
                cycle.calculate_abundance()
        except Exception as e:
            e.add_note(
                "This error occurred while running calculate_abundance() for cycle "
                f"{cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_smoothed_abundance(
            self, 
            constant_value: float = 1.0,
            constant_value_bounds: Tuple[float, float] = (0.1, 10),
            length_scale: float = 10.0,
            length_scale_bounds: Tuple[float, float] = (1.0, 200.0),
            alpha: float = 1.0,
            alpha_bounds: Tuple[float, float] = (0.1, 1e7),
            noise_level: float = 0.001,
            noise_level_bounds: Tuple[float, float] = (1e-4, 1.0),
            gp_alpha: float = 1e-10,
            n_restarts: int = 1,
            random_seed: int = 42
        ) -> Self:
        """
        Apply Gaussian process smoothing to abundance estimates for all 
        cell cycles in the experiment individually.

        Parameters
        ----------
        constant_value : float, optional
            Initial value for the constant kernel. Default is 1.0.
        constant_value_bounds : Tuple[float, float], optional
            Bounds for the constant kernel value. Default is (0.1, 10).
        length_scale : float, optional
            Initial length scale for the Rational Quadratic kernel. 
            Default is 10.0.
        length_scale_bounds : Tuple[float, float], optional
            Bounds for the length scale of the Rational Quadratic 
            kernel. Default is (1.0, 200.0).
        alpha : float, optional
            Initial alpha value for the Rational Quadratic kernel, which
            determines the relative weighting of large-scale and 
            small-scale variations. Default is 1.0.
        alpha_bounds : Tuple[float, float], optional
            Bounds for the alpha parameter of the Rational Quadratic 
            kernel. Default is (0.1, 1e7).
        noise_level : float, optional
            Initial noise level for the White kernel. Default is 0.001.
        noise_level_bounds : Tuple[float, float], optional
            Bounds for the noise level of the White kernel. 
            Default is (1e-4, 1.0).
        gp_alpha : float, optional
            Value added to the diagonal of the kernel matrix during 
            fitting to improve numerical stability. Default is 1e-10.
        n_restarts : int, optional
            Number of restarts for the optimizer to find the best kernel
            parameters. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.
        
        Returns
        -------
        Self
        """
        try:
            for cycle in self:
                cycle.calculate_smoothed_abundance(
                    constant_value,
                    constant_value_bounds,
                    length_scale,
                    length_scale_bounds,
                    alpha,
                    alpha_bounds,
                    noise_level,
                    noise_level_bounds,
                    gp_alpha,
                    n_restarts, 
                    random_seed
                )
        except Exception as e:
            e.add_note(
                "This error occurred while running calculate_smoothed_abundance() for "
                f"cycle {cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_production_rate(
            self, 
            apply_maturation_correction: bool = False,
            maturation_time: Optional[float] = None
        ) -> Self:
        """
        Calculate rate of change of fluorescent protein abundance for 
        all cell cycles in the experiment individually. Optionally, it 
        can apply a correction for the fluorophore maturation time.

        Parameters
        ----------
        apply_maturation_correction : bool, optional
            If True, applies a correction for the fluorophore 
            maturation time. Default is False.
        maturation_time : float, optional
            The maturation time of the fluorophore, required if 
            `apply_maturation_correction` is True.

        Returns
        -------
        Self
        """
        try:
            for cycle in self:
                cycle.calculate_production_rate(apply_maturation_correction, maturation_time)
        except Exception as e:
            e.add_note(
                "This error occurred while running calculate_production_rate() for "
                f"cycle {cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_smoothed_volume(
            self,
            constant_value: float = 10.0,
            constant_value_bounds: Tuple[float, float] = (1.0, 1000.0),
            length_scale: float = 200.0,
            length_scale_bounds: Tuple[float, float] = (10.0, 1000.0),
            noise_level: float = 1.0,
            noise_level_bounds: Tuple[float, float] = (0.01, 100.0),
            gp_alpha: float = 1e-10,
            n_restarts: int = 1, 
            random_seed: int = 42
        ) -> Self:
        """
        Apply Gaussian process smoothing to volume estimates for all 
        cell cycles in the experiment individually.

        Parameters
        ----------
        constant_value : float, optional
            Initial value for the constant kernel. Default is 10.0.
        constant_value_bounds : Tuple[float, float], optional
            Bounds for the constant kernel value. 
            Default is (1.0, 1000.0).
        length_scale : float, optional
            Initial length scale for the RBF kernel. Default is 200.0.
        length_scale_bounds : Tuple[float, float], optional
            Bounds for the RBF kernel length scale. 
            Default is (10.0, 1000.0).
        noise_level : float, optional
            Initial noise level for the White kernel. Default is 1.0.
        noise_level_bounds : Tuple[float, float], optional
            Bounds for the White kernel noise level. 
            Default is (0.01, 100.0).
        gp_alpha : float, optional
            Value added to the diagonal of the kernel matrix during 
            fitting to improve numerical stability. Default is 1e-10.
        n_restarts : int, optional
            Number of restarts for the optimizer to find the best kernel
            parameters. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        Self
        """
        try:
            for cycle in self:
                cycle.calculate_smoothed_volume(
                    constant_value,
                    constant_value_bounds,
                    length_scale,
                    length_scale_bounds,
                    noise_level,
                    noise_level_bounds,
                    gp_alpha,
                    n_restarts, 
                    random_seed
                )
        except Exception as e:
            e.add_note(
                "This error occurred while running calculate_smoothed_volume() for "
                f"cycle {cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_volume_specific_production_rate(self) -> Self:
        """
        Calculate volume-specific rate of change for all cell cycles
        in the experiment individually.

        Returns
        -------
        Self
        """
        try:
            for cycle in self.cell_cycles.values():
                cycle.calculate_volume_specific_production_rate()
        except Exception as e:
            e.add_note(
                "This error occurred while running "
                f"calculate_volume_specific_production_rate() for cycle "
                f"{cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_all_cycle_values(
            self,
            calculate_smoothed_abundance_kwargs: Dict[str, Any] = {},
            calculate_production_rate_kwargs: Dict[str, Any] = {},
            calculate_smoothed_volume_kwargs: Dict[str, Any] = {}
        ) -> Self:
        """
        Convenience method to run the complete analysis pipeline on all
        cell cycles in the experiment individually. Run time may be 
        long.

        Parameters
        ----------
        calculate_smoothed_abundance_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_smoothed_abundance` method.
            Default is an empty dictionary.
        calculate_production_rate_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_production_rate` method.
            Default is an empty dictionary.
        calculate_smoothed_volume_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to the 
            `calculate_smoothed_volume` method. 
            Default is an empty dictionary.

        Returns
        -------
        Self
        """
        # Check that maturation_time is provided if maturation correction is enabled.
        # Do it here so that we can raise the error before starting the analysis.
        if (
            calculate_production_rate_kwargs.get("apply_maturation_correction", False) 
            and calculate_production_rate_kwargs.get("maturation_time", None) is None
        ):
            raise ValueError(
                "Maturation time must be provided for maturation correction."
            )
        try:
            for cycle in self:
                cycle.calculate_all_cycle_values(
                    self.image_capture_interval,
                    self.max_extra_data_points,
                    calculate_smoothed_abundance_kwargs,
                    calculate_production_rate_kwargs,
                    calculate_smoothed_volume_kwargs
                )
        except Exception as e:
            e.add_note(
                "This error occurred while running calculate_all_cycle_values() for "
                f"cycle {cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def align_to_standard_coordinate(self) -> Self:
        """
        Map all cell cycles to standardized cell cycle progression 
        coordinates.
        """
        try:
            for cycle in self:
                cycle.align_to_standard_coordinate(self.standard_coordinate_anchors)
        except Exception as e:
            e.add_note(
                "This error occurred while running align_to_standard_coordinate() for "
                f"cycle {cycle.cycle_id} in experiment {self.experiment_id}."
            )
            raise e
        return self
    
    def calculate_standard_coordinate_anchors(
            self,
            cell_cycle_anchors: Sequence[str],
            standard_coordinate_begin: float = 0.0,
            standard_coordinate_end: float = 1.0
            ) -> Self:
        """
        Calculate standard coordinate anchor values based on the average
        fractions of a cell cycle at which the specified events occur.

        Assumes that:
        - The cell_cycle_anchors are provided in reasonable
        biological order.
        - That the first event should have the beginning coordinate.
        - That the last event should have the end coordinate.

        Parameters
        ----------
        cell_cycle_anchors : Sequence[str]
            A sequence of event names that define the anchors for the
            standard coordinate system. These events should be present
            in all cell cycles.
        standard_coordinate_begin : float, optional
            The value representing the start of the standard coordinate
            range. Default is 0.0.
        standard_coordinate_end : float, optional
            The value representing the end of the standard coordinate
            range. Default is 1.0.
        
        Returns
        -------
        Self
        """
        if len(cell_cycle_anchors) < 2:
            raise ValueError(
                "At least two cell cycle anchors are required to define standard "
                "coordinates."
            )
        # Get the time IDs for the specified cell cycle anchors across all cycles.
        time_id_matrix = np.empty(
            (len(self.cell_cycles), len(cell_cycle_anchors)), dtype=float
        )
        for i, cycle in enumerate(self.cell_cycles.values()):
            time_id_matrix[i, :] = cycle.get_time_ids_for_events(cell_cycle_anchors)

        # Remap to the standard coordinate range and find the mean time for each 
        # anchor event.
        # Reshape methods are required to ensure that the operations can broadcast
        # correctly across the time_id_matrix.
        cycle_min_time_id = time_id_matrix[:, 0].reshape(-1, 1)
        cycle_max_time_id = time_id_matrix[:, -1].reshape(-1, 1)
        mean_cycle_fractions = (
            (time_id_matrix - cycle_min_time_id) / (cycle_max_time_id - cycle_min_time_id) 
            * (standard_coordinate_end - standard_coordinate_begin)
            + standard_coordinate_begin
        ).mean(axis=0)
        standard_coordinates = {
            event : mean_cycle_fractions[i] for i, event in enumerate(cell_cycle_anchors)
        }
        # Store the calculated anchors.
        self._standard_coordinate_anchors = standard_coordinates
        return self

    def extract_aligned_cycle_data(self) -> Self:
        """
        Extract aligned cycle data into a single DataFrame.

        Returns
        -------
        Self
        """
        aligned_cycle_data = {
            cycle_id : cycle.extract_aligned_cycle_data()
            for cycle_id, cycle in self.cell_cycles.items()
        }
        self._aligned_cycle_data = pd.concat(aligned_cycle_data)
        return self
    
    def fit_production_rate_gp(
            self,
            rate_type: str,
            constant_value: float = 1.0,
            constant_value_bounds: Tuple[float, float] = (1e-5, 1e5),
            length_scale: float = 0.2,
            length_scale_bounds: Tuple[float, float] = (0.01, 1),
            noise_level: float = 1.0,
            noise_level_bounds: Tuple[float, float] = (1e-4, 1e4),
            gp_alpha: float = 1e-10,
            n_restarts: int = 1,
            random_seed: int = 42
        ) -> Self:
        """
        This method combines production rate data from all analyzed cell
        cycles and fits a single Gaussian process model to capture the 
        mean behavior across the standardized cell cycle progression.

        Parameters
        ----------
        rate_type : str, optional
            The type of production rate to fit. Supported values are:
            - "basic": Fit a Gaussian process to the estimated
                production rates.
            - "normalised": Fit a Gaussian process to the production
                rates normalised by the mean production rate on a per-
                cycle basis.
            - "specific": Fit a Gaussian process to the volume-specific
                production rates.
        constant_value : float, optional
            Initial value for the constant kernel. Default is 1.0.
        constant_value_bounds : Tuple[float, float], optional
            Bounds for the constant kernel value. 
            Default is (1e-5, 1e5).
        length_scale : float, optional
            Initial length scale for the RBF kernel. Default is 0.2.
        length_scale_bounds : Tuple[float, float], optional
            Bounds for the length scale of the RBF kernel. 
            Default is (0.01, 1).
        noise_level : float, optional
            Initial noise level for the White kernel. Default is 1.0.
        noise_level_bounds : Tuple[float, float], optional
            Bounds for the noise level of the White kernel. 
            Default is (1e-4, 1e4).
        gp_alpha : float, optional
            Value added to the diagonal of the kernel matrix during 
            fitting to improve numerical stability. Default is 1e-10.
        n_restarts : int, optional
            Number of restarts for the optimizer to find the best kernel
            parameters. Default is 1.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.
        Returns
        -------
        Self
        """
        # Perform smoothing using a composite Gaussian process kernel. The Constant
        # kernel captures the overall scale, The RBF kernel captures 
        # the smooth longer-term dynamic trends, while the White kernel approximates 
        # short term noise.
        constant_kernel = ConstantKernel(constant_value, constant_value_bounds)
        rbf_kernel = RBF(length_scale, length_scale_bounds)
        white_kernel = WhiteKernel(noise_level, noise_level_bounds)
        gp_kernel = constant_kernel * rbf_kernel + white_kernel
        gp_regressor = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=gp_alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
            random_state=random_seed
        )

        gp_time = self.aligned_cycle_data["Standard coordinate"].values[:, np.newaxis]
        match rate_type.strip().lower():
            case "basic":
                gp_rate = (
                    self.aligned_cycle_data["Production rate"].values[:, np.newaxis]
                )
            case "normalised":
                gp_rate = (
                    self.aligned_cycle_data["Normalised production rate"]
                    .values[:, np.newaxis]
                )
            case "specific":
                gp_rate = (
                    self.aligned_cycle_data["Specific production rate"]
                    .values[:, np.newaxis]
                )
            case _:
                raise ValueError(
                    f"Invalid rate_type '{rate_type}'. "
                    "Valid options are: 'basic', 'normalised', 'specific'."
                )
        mean_rate = gp_rate.mean()
        # Scale and center the data around 0 to produce better fits.
        gp_fit = gp_regressor.fit(gp_time, (gp_rate / mean_rate) - 1)

        # Store the Gaussian process model in the experiment.
        match rate_type.strip().lower():
            case "basic":
                self._production_rate_gp = gp_fit
            case "normalised":
                self._normalised_production_rate_gp = gp_fit
            case "specific":
                self._specific_production_rate_gp = gp_fit
        return self


    # Plotting methods.
    def plot_production_rate(
            self, rate_type: str, figsize: Tuple[float, float] = (10.0, 6.0)
        ) -> Figure:
        """
        Plot one of three different production rates across all cell 
        cycles in the experiment, aligned to a standard cell cycle
        coordinate system.

        Parameters
        ----------
        rate_type : str
            The type of production rate to plot. Supported values are:
            - "basic": Plot the estimated production rates.
            - "normalised": Plot the normalised production rates.
            - "specific": Plot the volume-specific production rates.
        figsize : Tuple[float, float], optional
            Size of the figure to create. Default is (10.0, 6.0).

        Returns
        -------
        Figure
            A matplotlib Figure object containing the plot.
        """
        # Calculate the mean and standard deviation from the Gaussian process model.
        # Do this first to ensure to raise an error as soon as possible if the
        # Gaussian process model has not been fitted.
        standard_coordinate_anchors = tuple(self.standard_coordinate_anchors.values())
        gp_time = np.linspace(
            standard_coordinate_anchors[0],
            standard_coordinate_anchors[-1],
            100
        )

        match rate_type.strip().lower():
            case "basic":
                gp_mean, gp_std = self.production_rate_gp.predict(
                    gp_time[:, np.newaxis], return_std=True
                )
                column = "Production rate"
                title = (
                    f"{self.experiment_id}, {len(self.cell_cycles)} cycles - "
                    "Production rate"
                )
                y_label = "Production rate (a.u. min⁻¹)"
            case "normalised":
                gp_mean, gp_std = self.normalised_production_rate_gp.predict(
                    gp_time[:, np.newaxis], return_std=True
                )
                column = "Normalised production rate"
                title = (
                    f"{self.experiment_id}, {len(self.cell_cycles)} cycles - "
                    "Normalised production Rate"
                )
                y_label = "Normalised production rate (a.u. min⁻¹)"
            case "specific":
                gp_mean, gp_std = self.specific_production_rate_gp.predict(
                    gp_time[:, np.newaxis], return_std=True
                )
                column = "Specific production rate"
                title = (
                    f"{self.experiment_id}, {len(self.cell_cycles)} cycles - "
                    "Volume specific production Rate"
                )
                y_label = "Volume specific production rate (a.u. min⁻¹ fL⁻¹)"
            case _:
                raise ValueError(
                    f"Invalid rate_type '{rate_type}'. "
                    "Valid options are: 'basic', 'normalised', 'specific'."
                )
        # Compensate for the scaling and centering performed before fitting.
        rate_mean = self.aligned_cycle_data[column].mean()
        gp_mean += 1
        gp_mean *= rate_mean
        gp_std *= rate_mean
        
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        # Plot the aligned cycle data.
        for cycle_id in self.cell_cycles:
            ax.plot(
                self.aligned_cycle_data.loc[cycle_id, "Standard coordinate"],
                self.aligned_cycle_data.loc[cycle_id, column],
                marker="o",
                linestyle="none",
                color="grey",
                alpha=0.5
            )
        # Plot the Gaussian process mean and standard deviation.
        ax.plot(
            gp_time, 
            gp_mean, 
            color="C0", 
            label="GP Mean"
        )
        ax.fill_between(
            gp_time,
            gp_mean - gp_std,
            gp_mean + gp_std,
            color="C0", 
            alpha=0.2, 
            label="GP St.Dev"
        )

        # Add lines indicating the positions of standard cell cycle coordinate anchors.
        styles = {
            f"{self.cycle_end_event}" : ("black", "-"),
            "Bud" : ("black", "--")
        }
        for i, (event, anchor) in enumerate(self.standard_coordinate_anchors.items()):
            # Cycle end event and Bud will have these appended, remove them.
            if event.endswith(("_0", "_1")):
                event_name = event[:-2]
            else:
                event_name = event
            # Add 4 here to hopefully maintain color equivalence with CellCycle
            # plotting methods.
            color, linestyle = styles.get(event_name, (f"C{i+4}", ":"))
            ax.axvline(anchor, color=color, linestyle=linestyle, label=event_name)

        ax.set_xlabel("Standard cell cycle coordinate")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(*_deduplicate_legend_items(ax.get_legend_handles_labels()))
        return fig


    # Additional methods
    def filter_cycles(
            self, 
            filter_func: Callable[..., bool],
            *args : Any,
            **kwargs: Any
        ) -> FluorescentProteinProductionRateExperiment:
        """
        Returns a new FluorescentProteinProductionRateExperiment
        containing only the cell cycles for which filter_func was True.

        Parameters
        ----------
        filter_func : Callable[..., bool]
            A callable which takes a CellCycle as it's first argument 
            and returns True if that CellCycle should be present in the 
            returned FluorescentProteinProductionRateExperiment. This
            callable may optionally accept additional arguments.
        args : Any 
            Optional positional arguments to be passed to filter_func.
        kwargs : Any
            Optional keyword arguments to be passed to filter_func.

        Returns
        -------
        filtered_experiment : FluorescentProteinProductionRateExperiment
            A new FluorescentProteinProductionRateExperiment
            containing only the cell cycles for which filter_func was 
            True.

        Notes
        -----
        CellCycles in the returned 
        FluorescentProteinProductionRateExperiment will keep any
        calculated values except for any standardised cell cycle
        coordinates. Calculation of standard cell cycle anchor
        coordinates, extraction of aligned data, and fitting of the 
        specific protein production rate Gaussian process will need to
        be performed (again) for the new 
        FluorescentProteinProductionRateExperiment.

        Examples
        --------
        >>> # Filter for cycles 90 min or longer.
        >>> filt_expt = expt.filter(lambda x: x.cycle_duration >= 90)
        >>>
        >>> # Filter for cycles longer than a value provided as a 
        >>> # parameter
        >>> def cycle_longer_than_t(cycle: CellCycle, t: float):
        >>>     return cycle.cycle_duration > t
        >>>
        >>> min_duration = 90
        >>> filt_expt = expt.filter(cycle_longer_than_t, min_duration)
        """
        keep_cycles = [cycle for cycle in self if filter_func(cycle, *args, **kwargs)]
        if not keep_cycles:
            raise ValueError("No cycles passed the filter!")
        
        filtered_experiment = FluorescentProteinProductionRateExperiment(
            self.experiment_id,
            self.image_capture_interval,
            self.cycle_end_event,
            self.min_extra_data_points,
            self.max_extra_data_points
        )
        # Deepcopy to ensure that changes to CellCycle instances beloging to one
        # FluorescentProteinProductionRateExperiment instance do not propagate to 
        # others in unexpected ways.
        filtered_experiment._cell_cycles = {
            cycle.cycle_id : copy.deepcopy(cycle) for cycle in keep_cycles
        }
        # Standard coordinates are no longer valid if the set of cycles
        # for which they were calculated has changed so remove them.
        for cycle in filtered_experiment:
            cycle._drop_column("Standard coordinate")
        return filtered_experiment


def get_version() -> Dict[str, str]:
    """
    Returns the module version and the SHA256 hash of this file.
    """
    version_info = {}
    version_info["version"] = _VERSION
    source_code_path = inspect.getabsfile(inspect.currentframe())
    with open(source_code_path, "rb") as f:
        hash = hashlib.file_digest(f, "sha256")
    version_info["SHA256"] = hash.hexdigest()
    return version_info


def _deduplicate_legend_items(
            items: Tuple[List[Artist], List[Any]]
        ) -> Tuple[List[Artist], List[Any]]:
        """Remove any legend entries with duplicated labels."""
        new_handles = []
        new_labels = []
        for handle, label in zip(*items):
            if not label in new_labels:
                new_handles.append(handle)
                new_labels.append(label)
        return new_handles, new_labels