"""Reusable feature engineering for stellar classification experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stellar.constants import CATEGORICAL_COLS, MAG_COLS


SDSS_FILTER_WAVELENGTHS_ANGSTROM = {
    "u": 3543.0,
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
    "z": 9134.0,
}


def _clip_positive(values: pd.Series) -> pd.Series:
    return values.clip(lower=0.0)


def add_astronomy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable color, magnitude, redshift, and angle-cycle features."""
    df = df.copy()

    if all(col in df.columns for col in MAG_COLS):
        adjacent_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
        wider_pairs = [("u", "r"), ("g", "i"), ("r", "z"), ("u", "z")]
        for left, right in adjacent_pairs + wider_pairs:
            df[f"{left}_minus_{right}"] = df[left] - df[right]

        mag_values = df[MAG_COLS]
        df["mag_mean"] = mag_values.mean(axis=1)
        df["mag_std"] = mag_values.std(axis=1)
        df["mag_min"] = mag_values.min(axis=1)
        df["mag_max"] = mag_values.max(axis=1)
        df["mag_range"] = df["mag_max"] - df["mag_min"]

    if "redshift" in df.columns and all(col in df.columns for col in MAG_COLS):
        for col in MAG_COLS:
            df[f"{col}_redshift_ratio"] = df[col] / (df["redshift"].abs() + 1e-3)
        df["redshift_x_mag_mean"] = df["redshift"] * df["mag_mean"]

    if "alpha" in df.columns:
        alpha_rad = np.deg2rad(df["alpha"])
        df["alpha_sin"] = np.sin(alpha_rad)
        df["alpha_cos"] = np.cos(alpha_rad)

    if "delta" in df.columns:
        delta_rad = np.deg2rad(df["delta"])
        df["delta_sin"] = np.sin(delta_rad)
        df["delta_cos"] = np.cos(delta_rad)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def add_sdss_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SDSS filter-system and target-selection inspired color features."""
    df = df.copy()
    eps = 1e-3

    required_colors = ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]
    if all(col in df.columns for col in required_colors):
        gr = df["g_minus_r"]
        ri = df["r_minus_i"]
        iz = df["i_minus_z"]
        ug = df["u_minus_g"]

        df["sdss_lrg_c_perp"] = ri - gr / 4.0 - 0.18
        df["sdss_lrg_c_parallel"] = 0.7 * gr + 1.2 * (ri - 0.18)
        df["sdss_lrg_cut1_cperp_margin"] = 0.2 - df["sdss_lrg_c_perp"].abs()
        df["sdss_lrg_cut2_cperp_margin"] = df["sdss_lrg_c_perp"] - (0.45 - gr / 6.0)
        df["sdss_lrg_cut2_gr_margin"] = gr - (1.30 + 0.25 * ri)

        df["sdss_quasar_ugri_radius"] = np.sqrt(ug**2 + gr**2 + ri**2)
        df["sdss_quasar_griz_radius"] = np.sqrt(gr**2 + ri**2 + iz**2)
        df["sdss_quasar_blue_margin"] = 0.8 - ug

        df["sdss_red_dwarf_ri_over_1"] = _clip_positive(ri - 1.0)
        df["sdss_red_dwarf_ri_over_1_8"] = _clip_positive(ri - 1.8)
        df["sdss_carbon_star_gr_over_085"] = _clip_positive(gr - 0.85)
        df["sdss_carbon_star_ri_line_margin"] = (-0.4 + 0.65 * gr) - ri
        df["sdss_subdwarf_gr_over_16"] = _clip_positive(gr - 1.6)
        df["sdss_white_dwarf_blue_score"] = _clip_positive(-0.15 - gr) + _clip_positive(
            -(ug + 2.0 * gr)
        )

    if all(col in df.columns for col in MAG_COLS):
        log_lam = {
            col: np.log10(wavelength)
            for col, wavelength in SDSS_FILTER_WAVELENGTHS_ANGSTROM.items()
        }
        mag_slopes = {}
        for left, right in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
            name = f"sdss_mag_slope_{left}{right}"
            mag_slopes[name] = (df[left] - df[right]) / (log_lam[left] - log_lam[right])
            df[name] = mag_slopes[name]

        df["sdss_mag_slope_uz"] = (df["u"] - df["z"]) / (log_lam["u"] - log_lam["z"])
        df["sdss_sed_curvature_ugr"] = (
            df["sdss_mag_slope_ug"] - df["sdss_mag_slope_gr"]
        )
        df["sdss_sed_curvature_gri"] = (
            df["sdss_mag_slope_gr"] - df["sdss_mag_slope_ri"]
        )
        df["sdss_sed_curvature_riz"] = (
            df["sdss_mag_slope_ri"] - df["sdss_mag_slope_iz"]
        )
        df["sdss_flux_ratio_u_z"] = 10.0 ** (-0.4 * (df["u"] - df["z"]))
        df["sdss_flux_ratio_g_r"] = 10.0 ** (-0.4 * (df["g"] - df["r"]))
        df["sdss_flux_ratio_r_i"] = 10.0 ** (-0.4 * (df["r"] - df["i"]))

    if "redshift" in df.columns and "sdss_lrg_c_parallel" in df.columns:
        redshift_abs = df["redshift"].abs()
        df["sdss_cparallel_x_redshift"] = df["sdss_lrg_c_parallel"] * df["redshift"]
        df["sdss_cperp_over_redshift_abs"] = df["sdss_lrg_c_perp"] / (
            redshift_abs + eps
        )

    return df


def add_error_diagnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features aimed at known STAR/GALAXY and low-redshift hard slices."""
    df = df.copy()
    eps = 1e-3

    if "redshift" in df.columns:
        redshift = df["redshift"]
        redshift_abs = redshift.abs()
        df["redshift_low_00497_0127"] = (
            (redshift > 0.0497) & (redshift <= 0.127)
        ).astype("int8")
        df["redshift_very_low_le_00497"] = (redshift <= 0.0497).astype("int8")
        df["redshift_low_le_0144"] = (redshift <= 0.144).astype("int8")
        df["redshift_abs_low_00497_0127"] = (
            (redshift_abs > 0.0497) & (redshift_abs <= 0.127)
        ).astype("int8")
        df["redshift_abs_very_low_le_00497"] = (redshift_abs <= 0.0497).astype("int8")
        df["redshift_low_mid_distance"] = _clip_positive(redshift - 0.0497) * (
            redshift <= 0.127
        )
        df["redshift_low_mid_inverse"] = 1.0 / (redshift_abs + eps)

    if all(col in df.columns for col in ["spectral_type", "galaxy_population"]):
        spectral = df["spectral_type"].astype("string")
        population = df["galaxy_population"].astype("string")
        df["is_spectral_type_m"] = (spectral == "M").astype("int8")
        df["is_spectral_type_af"] = (spectral == "A/F").astype("int8")
        df["is_red_sequence"] = (population == "Red_Sequence").astype("int8")
        df["is_blue_cloud"] = (population == "Blue_Cloud").astype("int8")
        df["is_m_red_sequence"] = (
            df["is_spectral_type_m"] & df["is_red_sequence"]
        ).astype("int8")

    star_slice_cols = ["g_minus_r", "u_minus_z", "mag_range", "mag_std"]
    if all(col in df.columns for col in star_slice_cols):
        gr_high = _clip_positive(df["g_minus_r"] - 1.037)
        uz_high = _clip_positive(df["u_minus_z"] - 3.834)
        range_high = _clip_positive(df["mag_range"] - 3.949)
        std_high = _clip_positive(df["mag_std"] - 1.624)
        std_mid_high = _clip_positive(df["mag_std"] - 1.118)

        df["hard_star_gr_high"] = gr_high
        df["hard_star_uz_high"] = uz_high
        df["hard_star_mag_range_high"] = range_high
        df["hard_star_mag_std_high"] = std_high
        df["hard_star_mag_std_mid_high"] = std_mid_high
        df["hard_star_color_spread_score"] = (
            gr_high + uz_high + range_high + std_high
        )
        df["hard_star_color_spread_flag"] = (
            (df["g_minus_r"] > 1.037)
            & (df["u_minus_z"] > 3.834)
            & ((df["mag_range"] > 3.949) | (df["mag_std"] > 1.624))
        ).astype("int8")
        df["u_z_over_mag_range"] = df["u_minus_z"] / (df["mag_range"].abs() + eps)
        df["mag_std_over_mag_range"] = df["mag_std"] / (df["mag_range"].abs() + eps)

        if "is_spectral_type_m" in df.columns:
            is_m = df["is_spectral_type_m"].astype(float)
            df["m_x_hard_star_gr_high"] = is_m * gr_high
            df["m_x_hard_star_uz_high"] = is_m * uz_high
            df["m_x_hard_star_mag_range_high"] = is_m * range_high
            df["m_x_hard_star_mag_std_high"] = is_m * std_high
            df["m_x_hard_star_color_spread_score"] = (
                is_m * df["hard_star_color_spread_score"]
            )

        if "is_red_sequence" in df.columns:
            is_red = df["is_red_sequence"].astype(float)
            df["redseq_x_hard_star_color_spread_score"] = (
                is_red * df["hard_star_color_spread_score"]
            )

        if "redshift_low_00497_0127" in df.columns:
            low_mid = df["redshift_low_00497_0127"].astype(float)
            df["lowz_mid_x_g_minus_r"] = low_mid * df["g_minus_r"]
            df["lowz_mid_x_u_minus_z"] = low_mid * df["u_minus_z"]
            df["lowz_mid_x_mag_range"] = low_mid * df["mag_range"]
            df["lowz_mid_x_mag_std"] = low_mid * df["mag_std"]
            df["lowz_mid_x_hard_star_score"] = (
                low_mid * df["hard_star_color_spread_score"]
            )

        if "redshift_low_le_0144" in df.columns:
            low = df["redshift_low_le_0144"].astype(float)
            df["lowz_le0144_x_g_minus_r"] = low * df["g_minus_r"]
            df["lowz_le0144_x_u_minus_z"] = low * df["u_minus_z"]

    return df


def add_targeted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add external-knowledge and diagnostic features after base astronomy features."""
    df = add_sdss_external_features(df)
    df = add_error_diagnostic_features(df)
    return df


def add_local_guard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact low-redshift features for local STAR/GALAXY guards."""
    df = df.copy()
    eps = 1e-3

    color_cols = ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]
    if all(col in df.columns for col in color_cols):
        color_frame = df[color_cols]
        df["compact_color_std"] = color_frame.std(axis=1)
        df["compact_color_range"] = color_frame.max(axis=1) - color_frame.min(axis=1)
        df["compact_color_l1"] = color_frame.abs().sum(axis=1)
        df["compact_color_score"] = 1.0 / (1.0 + df["compact_color_std"].abs())
        df["blue_compact_score"] = df["compact_color_score"] * _clip_positive(
            0.9 - df["u_minus_g"]
        )

        if "g_minus_r" in df.columns and "r_minus_i" in df.columns:
            df["compact_gr_ri_absdiff"] = (df["g_minus_r"] - df["r_minus_i"]).abs()
            df["blue_cloud_color_margin"] = (
                _clip_positive(0.95 - df["u_minus_g"])
                + _clip_positive(0.55 - df["g_minus_r"])
                + _clip_positive(0.35 - df["r_minus_i"])
            )

    if "redshift" in df.columns:
        redshift_abs = df["redshift"].abs()
        df["local_redshift_abs"] = redshift_abs
        df["local_lowz_le_008"] = (redshift_abs <= 0.08).astype("int8")
        df["local_lowz_le_012"] = (redshift_abs <= 0.12).astype("int8")
        df["local_lowz_le_016"] = (redshift_abs <= 0.16).astype("int8")
        df["local_lowz_window_004_016"] = (
            redshift_abs.between(0.04, 0.16, inclusive="both")
        ).astype("int8")

    if "mag_std" in df.columns:
        df["low_mag_std_score"] = 1.0 / (1.0 + df["mag_std"].abs())
        df["low_mag_std_le_060"] = (df["mag_std"] <= 0.60).astype("int8")
        df["low_mag_std_le_080"] = (df["mag_std"] <= 0.80).astype("int8")
        df["low_mag_std_le_100"] = (df["mag_std"] <= 1.00).astype("int8")

    if all(col in df.columns for col in ["local_redshift_abs", "compact_color_score"]):
        df["lowz_x_compact_color"] = (
            (1.0 / (df["local_redshift_abs"] + eps)) * df["compact_color_score"]
        )

    if all(col in df.columns for col in ["is_blue_cloud", "compact_color_score"]):
        df["blue_cloud_x_compact_color"] = (
            df["is_blue_cloud"].astype(float) * df["compact_color_score"]
        )

    if all(col in df.columns for col in ["is_blue_cloud", "low_mag_std_score"]):
        df["blue_cloud_x_low_mag_std"] = (
            df["is_blue_cloud"].astype(float) * df["low_mag_std_score"]
        )

    return df
