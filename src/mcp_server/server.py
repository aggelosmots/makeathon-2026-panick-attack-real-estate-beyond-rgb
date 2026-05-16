from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import math
import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from src.common_config import DATA_ROOT as CONFIGURED_DATA_ROOT
from src.common_config import env_bool, env_int, env_str

MCP_HOST = env_str("MCP_HOST", "0.0.0.0")
MCP_PORT = env_int("MCP_PORT", 8000)
ALLOW_WRITE_TO_DATA = env_bool("ALLOW_WRITE_TO_DATA", True)
LOCAL_DATA_ROOT = Path("data").resolve()
DATA_ROOT = LOCAL_DATA_ROOT if not CONFIGURED_DATA_ROOT.exists() and LOCAL_DATA_ROOT.exists() else CONFIGURED_DATA_ROOT
MCP_TOOL_OUTPUT_LOG_PATH = DATA_ROOT / "mcp_tool_outputs.jsonl"
SOIL_PLOT_SPECS = {
    "ph_profile": {
        "column": "pH_Assessment",
        "title": "Soil pH Profile",
        "xlabel": "pH",
        "default_color": "#4f81bd",
        "mean_label_format": "Mean: {:.2f}",
    },
    "nitrogen_profile": {
        "column": "Nitrogen_N_mg_kg",
        "title": "Total Nitrogen (N)",
        "xlabel": "Concentration (mg/kg)",
        "default_color": "#c0504d",
        "mean_label_format": "Mean: {:.1f} mg/kg",
    },
    "phosphorus_profile": {
        "column": "Phosphorus_P_mg_kg",
        "title": "Available Phosphorus (P)",
        "xlabel": "Concentration (mg/kg)",
        "default_color": "#9bbb59",
        "mean_label_format": "Mean: {:.1f} mg/kg",
    },
    "potassium_profile": {
        "column": "Potassium_K_mg_kg",
        "title": "Exchangeable Potassium (K)",
        "xlabel": "Concentration (mg/kg)",
        "default_color": "#8064a2",
        "mean_label_format": "Mean: {:.1f} mg/kg",
    },
    "magnesium_profile": {
        "column": "Magnesium_Mg_mg_kg",
        "title": "Magnesium (Mg)",
        "xlabel": "Concentration (mg/kg)",
        "default_color": "#4bacc6",
        "mean_label_format": "Mean: {:.1f} mg/kg",
    },
    "som_profile": {
        "column": "SOM_pct",
        "title": "Soil Organic Matter (SOM %)",
        "xlabel": "Organic Matter (%)",
        "default_color": "#f79646",
        "mean_label_format": "Mean: {:.1f}%",
    },
}
TEXTBOX_FIELDS = [
    ("pH_Assessment", "Mean Soil pH", "{:.2f}"),
    ("SOM_pct", "Mean SOM", "{:.2f}%"),
    ("SOC_pct", "Mean SOC", "{:.2f}%"),
    ("Nitrogen_N_mg_kg", "Mean Nitrogen (N)", "{:.1f} mg/kg"),
    ("Phosphorus_P_mg_kg", "Mean Phosphorus (P)", "{:.1f} mg/kg"),
    ("Potassium_K_mg_kg", "Mean Potassium (K)", "{:.1f} mg/kg"),
    ("Magnesium_Mg_mg_kg", "Mean Magnesium (Mg)", "{:.1f} mg/kg"),
    ("NDVI", "Mean NDVI", "{:.3f}"),
    ("SWI", "Mean SWI", "{:.3f}"),
    ("Visible_Reflectance_Mean", "Visible Reflectance", "{:.4f}"),
    ("NIR_Reflectance_Mean", "NIR Reflectance", "{:.4f}"),
    ("SWIR1_Reflectance_Mean", "SWIR1 Reflectance", "{:.4f}"),
    ("SWIR2_Reflectance_Mean", "SWIR2 Reflectance", "{:.4f}"),
]
REPORTS_DIR = "reports"

mcp = FastMCP(
    "local-data-toolset",
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path="/mcp",
)


def _append_mcp_tool_output_log(tool_name: str, arguments: dict[str, Any], output: Any) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "output": output,
    }
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        with MCP_TOOL_OUTPUT_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def logged_mcp_tool(func):
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        try:
            output = func(*args, **kwargs)
        except Exception as exc:
            _append_mcp_tool_output_log(
                func.__name__,
                arguments,
                {"success": False, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        _append_mcp_tool_output_log(func.__name__, arguments, output)
        return output

    wrapper.__signature__ = signature
    return mcp.tool()(wrapper)


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Optional dependency `{module_name}` is required for this tool but is not installed."
        ) from exc


def _rasterio():
    return _optional_import("rasterio")


def _tool_error(name: str, path: str, exc: Exception) -> dict[str, Any]:
    return {"success": False, "plot_name": name, "path": path, "error": f"{type(exc).__name__}: {exc}"}


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "report"


def _safe_path(relative_path: str | None = ".") -> Path:
    """Resolve a user path under DATA_ROOT and block path traversal."""
    relative_path = relative_path or "."
    candidate = (DATA_ROOT / relative_path).resolve()
    data_root = DATA_ROOT.resolve()

    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError(f"Path escapes DATA_ROOT: {relative_path}")

    return candidate


def _resolve_data_path(path_text: str | None) -> Path:
    """Accept a DATA_ROOT-relative path, container path, or copied Windows host path."""
    if not path_text:
        raise ValueError("Path is required.")

    raw_path = str(path_text).strip().strip("\"'")
    data_root = DATA_ROOT.resolve()

    if "\\" in raw_path or ":" in raw_path:
        windows_parts = PureWindowsPath(raw_path).parts
        lowered_parts = [part.lower() for part in windows_parts]
        if "data" in lowered_parts:
            data_index = lowered_parts.index("data")
            relative_parts = windows_parts[data_index + 1:]
            if relative_parts:
                return _safe_path(str(Path(*relative_parts)))

    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved == data_root or data_root in resolved.parents:
            return resolved
        raise ValueError(f"Path is outside DATA_ROOT. Use a path under {DATA_ROOT} or a path relative to it.")

    return _safe_path(raw_path)


def _resolve_tif_path(path_text: str | None) -> Path:
    target_path = _resolve_data_path(path_text)

    if target_path.is_dir():
        spectral_candidate = target_path / "SPECTRAL_IMAGE.TIF"
        if spectral_candidate.exists() and spectral_candidate.is_file():
            return spectral_candidate

        tif_candidates = sorted(
            list(target_path.glob("*.TIF"))
            + list(target_path.glob("*.tif"))
            + list(target_path.glob("*.TIFF"))
            + list(target_path.glob("*.tiff"))
        )
        if tif_candidates:
            return tif_candidates[0]

        raise ValueError(f"No GeoTIFF files were found under: {target_path}")

    return target_path


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "relative_path": str(path.resolve().relative_to(DATA_ROOT.resolve())),
        "type": "dir" if path.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified_epoch": int(stat.st_mtime),
    }


def _discover_tif_paths() -> list[str]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return [
        str(path.relative_to(DATA_ROOT.resolve()))
        for path in sorted(DATA_ROOT.glob("*"))
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]


def _clean_float(value: Any, digits: int = 4) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _scale_reflectance(data: np.ndarray) -> np.ndarray:
    arr = data.astype("float32", copy=False)
    arr[arr == 0] = np.nan
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanpercentile(finite, 99)) > 2.0:
        arr = arr / 10000.0
    return np.clip(arr, 0.0, 1.0)

def _safe_band(src: Any, band_number: int) -> np.ndarray | None:
    if band_number < 1 or band_number > src.count:
        return None
    return _scale_reflectance(src.read(band_number))

def _mean_bands(src: Any, start: int, end: int) -> np.ndarray | None:
    band_numbers = [band for band in range(start, end + 1) if 1 <= band <= src.count]
    if not band_numbers:
        return None

    arrays = [_scale_reflectance(src.read(band)) for band in band_numbers]
    return np.nanmean(np.stack(arrays), axis=0)

def _array_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "valid_pixels": 0,
        }

    return {
        "mean": _clean_float(np.nanmean(finite)),
        "std": _clean_float(np.nanstd(finite)),
        "min": _clean_float(np.nanmin(finite)),
        "max": _clean_float(np.nanmax(finite)),
        "valid_pixels": int(finite.size),
    }

def _center_coordinates(src: Any) -> dict[str, Any]:
    bounds = src.bounds
    center_x = (bounds.left + bounds.right) / 2.0
    center_y = (bounds.bottom + bounds.top) / 2.0

    native = {
        "x": _clean_float(center_x, 6),
        "y": _clean_float(center_y, 6),
    }
    crs_text = src.crs.to_string() if src.crs else None

    if src.crs and not src.crs.is_geographic:
        try:
            pyproj = _optional_import("pyproj")
            transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(center_x, center_y)
            return {
                "lat": _clean_float(lat, 6),
                "lon": _clean_float(lon, 6),
                "native": native,
                "crs": crs_text,
            }
        except Exception as exc:
            return {
                "lat": None,
                "lon": None,
                "native": native,
                "crs": crs_text,
                "warning": f"Could not transform projected CRS to WGS84: {exc}",
            }

    return {
        "lat": _clean_float(center_y, 6),
        "lon": _clean_float(center_x, 6),
        "native": native,
        "crs": crs_text,
    }


def _geotiff_metadata(src: Any, tif_path: Path) -> dict[str, Any]:
    return {
        "path": str(tif_path.relative_to(DATA_ROOT.resolve())),
        "name": tif_path.name,
        "width": int(src.width),
        "height": int(src.height),
        "bands": int(src.count),
        "crs": src.crs.to_string() if src.crs else None,
        "bounds": {
            "west": _clean_float(src.bounds.left, 4),
            "south": _clean_float(src.bounds.bottom, 4),
            "east": _clean_float(src.bounds.right, 4),
            "north": _clean_float(src.bounds.top, 4),
        },
        "center": _center_coordinates(src),
    }



def _analyze_tif(path_text: str) -> dict[str, Any]:
    rasterio = _rasterio()
    tif_path = _resolve_tif_path(path_text)
    
    if not tif_path.exists():
        raise ValueError(f"GeoTIFF file does not exist: {path_text}")
    if not tif_path.is_file():
        raise ValueError(f"GeoTIFF path is not a file: {path_text}")

    with rasterio.open(tif_path) as src:

        # extract specific bands
        red  = _safe_band(src, 40)
        nir  = _safe_band(src, 80)
        swir = _safe_band(src, 200)

        if red is None or nir is None:
            raise ValueError(
                f"{tif_path.name} has {src.count} bands; NDVI requires at least bands 40 and 80."
            )

        # compute additional indeces
        ndvi = (nir - red) / (nir + red + 1e-6)
        healthy_mask = ndvi >= 0.35
        stressed_mask = ndvi < 0.2
        ndvi_stats = _array_stats(ndvi)
        valid_ndvi = ndvi[np.isfinite(ndvi)]
        valid_pixels = int(valid_ndvi.size)

        healthy_coverage = None
        stressed_coverage = None
        if valid_pixels:
            healthy_coverage = round(float(np.count_nonzero(healthy_mask & np.isfinite(ndvi))) / valid_pixels * 100.0, 2)
            stressed_coverage = round(float(np.count_nonzero(stressed_mask & np.isfinite(ndvi))) / valid_pixels * 100.0, 2)

        swir_stats = _array_stats(swir) if swir is not None else None
        moisture_index = None
        if swir is not None:
            moisture = nir / (swir + 1e-6)
            moisture_index = _array_stats(moisture)

        # stats
        visible_mean = _mean_bands(src, 1, 40)
        nir_mean = _mean_bands(src, 45, 85)
        swir_mean = _mean_bands(src, 150, min(220, src.count))

        score = None
        if ndvi_stats["mean"] is not None and ndvi_stats["std"] is not None and healthy_coverage is not None:
            # Bounded, transparent score: vegetation health + coverage - field variability penalty.
            score = round(
                (float(ndvi_stats["mean"]) * 55.0)
                + (healthy_coverage / 100.0 * 35.0)
                + (max(0.0, 1.0 - float(ndvi_stats["std"])) * 10.0),
                3,
            )

        raster_metadata = _geotiff_metadata(src, tif_path)
        return {
            "path": raster_metadata["path"],
            "name": raster_metadata["name"],
            "metadata": {key: value for key, value in raster_metadata.items() if key not in {"path", "name"}},
            "metrics": {
                "mean_ndvi": ndvi_stats["mean"],
                "ndvi_std": ndvi_stats["std"],
                "ndvi_min": ndvi_stats["min"],
                "ndvi_max": ndvi_stats["max"],
                "healthy_coverage_pct": healthy_coverage,
                "stressed_coverage_pct": stressed_coverage,
                "valid_pixels": valid_pixels,
                "swir_band_200": swir_stats,
                "moisture_index_nir_over_swir": moisture_index,
                "visible_reflectance_mean": _clean_float(np.nanmean(visible_mean)) if visible_mean is not None else None,
                "nir_reflectance_mean": _clean_float(np.nanmean(nir_mean)) if nir_mean is not None else None,
                "swir_reflectance_mean": _clean_float(np.nanmean(swir_mean)) if swir_mean is not None else None,
                "investment_score": score,
            },
            "methodology": {
                "reflectance_scaling": "Bands are divided by 10000 when high reflectance scale is detected; zero mosaic edges are ignored as nodata.",
                "ndvi_formula": "(Band_80 - Band_40) / (Band_80 + Band_40)",
                "healthy_coverage_threshold": "NDVI >= 0.35",
                "stressed_coverage_threshold": "NDVI < 0.20",
                "score_formula": "mean_ndvi*55 + healthy_coverage_fraction*35 + (1-ndvi_std)*10",
            },
        }


def _plotting_modules() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    matplotlib = _optional_import("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    return pyplot


def _tif_to_dataframe(tif_path: str, max_pixels: int = 250000) -> pd.DataFrame:
    """Convert a hyperspectral GeoTIFF into a sampled band dataframe for plotting."""
    rasterio = _rasterio()
    resolved_tif = _resolve_tif_path(tif_path)
    if not resolved_tif.exists():
        raise ValueError(f"GeoTIFF file does not exist: {tif_path}")
    if not resolved_tif.is_file():
        raise ValueError(f"GeoTIFF path is not a file: {tif_path}")

    with rasterio.open(resolved_tif) as src:
        num_pixels = int(src.height * src.width)
        sample_count = min(max(1, int(max_pixels)), num_pixels)
        if sample_count < num_pixels:
            indices = np.linspace(0, num_pixels - 1, sample_count, dtype=np.int64)
        else:
            indices = np.arange(num_pixels, dtype=np.int64)

        band_data: dict[str, Any] = {}
        for band_number in range(1, src.count + 1):
            band = _scale_reflectance(src.read(band_number)).reshape(-1)[indices]
            band_data[f"Band_{band_number}"] = band

    df = pd.DataFrame(band_data)
    df.insert(0, "Pixel_ID", np.arange(len(df)))
    return df


def _require_bands(df: pd.DataFrame, required_bands: list[int]) -> None:
    missing = [f"Band_{band}" for band in required_bands if f"Band_{band}" not in df.columns]
    if missing:
        raise ValueError(f"Required spectral bands are missing for this plot: {', '.join(missing)}")


def _band_mean(df: pd.DataFrame, start: int, end: int) -> pd.Series:
    columns = [f"Band_{band}" for band in range(start, end + 1) if f"Band_{band}" in df.columns]
    if not columns:
        raise ValueError(f"No bands are available in requested range Band_{start}..Band_{end}.")
    return df[columns].mean(axis=1)


def _load_table_data(path_text: str) -> tuple[Path, pd.DataFrame] | None:
    data_path = _resolve_data_path(path_text)
    if data_path.suffix.lower() != ".csv":
        return None
    return data_path, pd.read_csv(data_path)








def _process_enmap_soil_data(tif_path: str, max_pixels: int = 250000) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Build only directly measured table fields or directly computed spectral indices."""
    table_data = _load_table_data(tif_path)
    if table_data is not None:
        table_path, table_df = table_data
        numeric_columns = [
            column
            for column in table_df.columns
            if pd.api.types.is_numeric_dtype(table_df[column])
        ]
        metadata = {
            "source_type": "measured_table",
            "path": str(table_path.relative_to(DATA_ROOT.resolve())),
            "rows": int(len(table_df)),
            "numeric_columns": numeric_columns,
            "methodology": "Values are read directly from the provided CSV table. No synthetic soil values are generated.",
        }
        return "[MEASURED TABLE DATA]\nNo values were synthesized.\n", table_df, metadata

    df = _tif_to_dataframe(tif_path, max_pixels=max_pixels)
    _require_bands(df, [40, 80, 100, 200])

    band_columns = [column for column in df.columns if column.startswith("Band_")]
    df_cleaned = df.copy()
    df_cleaned[band_columns] = df_cleaned[band_columns].clip(lower=0.0, upper=1.0)

    visible_mean = _band_mean(df_cleaned, 1, 40)
    nir_mean = _band_mean(df_cleaned, 45, 85)
    swir1_mean = _band_mean(df_cleaned, 100, 150)
    swir2_mean = _band_mean(df_cleaned, 150, min(220, len(band_columns)))
    ndvi = (df_cleaned["Band_80"] - df_cleaned["Band_40"]) / (df_cleaned["Band_80"] + df_cleaned["Band_40"] + 1e-5)
    swi = df_cleaned["Band_200"] / (df_cleaned["Band_80"] + 1e-5)

    output_df = pd.DataFrame({
        "Pixel_ID": df_cleaned["Pixel_ID"],
        "NDVI": np.round(ndvi, 3),
        "SWI": np.round(swi, 3),
        "Visible_Reflectance_Mean": np.round(visible_mean, 4),
        "NIR_Reflectance_Mean": np.round(nir_mean, 4),
        "SWIR1_Reflectance_Mean": np.round(swir1_mean, 4),
        "SWIR2_Reflectance_Mean": np.round(swir2_mean, 4),
    })

    summary_stats = output_df.describe().transpose()
    report = (
        "[VERIFIED SPECTRAL METRICS]\n"
        f"Mean NDVI           : {summary_stats.loc['NDVI', 'mean']:.3f}\n"
        f"Mean SWI            : {summary_stats.loc['SWI', 'mean']:.3f}\n"
        f"Visible Reflectance : {summary_stats.loc['Visible_Reflectance_Mean', 'mean']:.4f}\n"
        f"NIR Reflectance     : {summary_stats.loc['NIR_Reflectance_Mean', 'mean']:.4f}\n"
        f"SWIR1 Reflectance   : {summary_stats.loc['SWIR1_Reflectance_Mean', 'mean']:.4f}\n"
        f"SWIR2 Reflectance   : {summary_stats.loc['SWIR2_Reflectance_Mean', 'mean']:.4f}\n"
    )
    metadata = {
        "source_type": "enmap_geotiff",
        "sampled_pixels": int(len(output_df)),
        "max_pixels": int(max_pixels),
        "methodology": (
            "NDVI, SWI, and reflectance means are computed directly from EnMap bands. "
            "pH, nutrients, SOM, and SOC are not generated from GeoTIFF data."
        ),
        "required_bands": ["Band_40", "Band_80", "Band_100", "Band_200"],
        "unavailable_without_measured_table": [
            "pH_Assessment",
            "Nitrogen_N_mg_kg",
            "Phosphorus_P_mg_kg",
            "Potassium_K_mg_kg",
            "Magnesium_Mg_mg_kg",
            "SOM_pct",
            "SOC_pct",
        ],
    }
    return report, output_df, metadata


def _apply_global_plot_styles() -> Any:
    pyplot = _plotting_modules()
    pyplot.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 16,
    })
    return pyplot


def _default_plot_path(tif_path: str, plot_name: str, image_format: str) -> str:
    stem = Path(str(tif_path).strip().strip("\"'")).stem or "parcel"
    clean_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    clean_plot = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in plot_name)
    return f"plots/{clean_stem}_{clean_plot}.{image_format.lower()}"


def _save_plot(fig: Any, tif_path: str, plot_name: str, output_path: str = "", image_format: str = "svg") -> dict[str, Any]:
    pyplot = _plotting_modules()
    normalized_format = image_format.lower().strip() or "svg"
    if normalized_format not in {"svg", "png", "pdf"}:
        raise ValueError("image_format must be one of: svg, png, pdf")

    target = _safe_path(output_path or _default_plot_path(tif_path, plot_name, normalized_format))
    if target.suffix.lower() != f".{normalized_format}":
        target = target.with_suffix(f".{normalized_format}")
    target.parent.mkdir(parents=True, exist_ok=True)


    fig.tight_layout()
    fig.savefig(target, format=normalized_format, bbox_inches="tight")
    pyplot.close(fig)
    return {
        "success": True,
        "relative_path": str(target.relative_to(DATA_ROOT.resolve())),
        "chart_type": normalized_format,
        "plot_name": plot_name,
    }


def _write_data_json(relative_path: str, payload: Any) -> dict[str, Any]:
    target = _safe_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "success": True,
        "relative_path": str(target.relative_to(DATA_ROOT.resolve())),
        "content_type": "application/json",
    }


def _write_data_markdown(relative_path: str, content: str) -> dict[str, Any]:
    target = _safe_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "success": True,
        "relative_path": str(target.relative_to(DATA_ROOT.resolve())),
        "content_type": "text/markdown",
    }


def _histogram_plot(
    *,
    tif_path: str,
    output_path: str,
    image_format: str,
    max_pixels: int,
    column: str,
    title: str,
    xlabel: str,
    color: str,
    mean_label_format: str,
    plot_name: str,
) -> dict[str, Any]:
    _, df, metadata = _process_enmap_soil_data(tif_path, max_pixels=max_pixels)
    if column not in df.columns:
        return {
            "success": False,
            "status": "insufficient_data",
            "plot_name": plot_name,
            "path": tif_path,
            "missing_fields": [column],
            "available_fields": list(df.columns),
            "metadata": metadata,
            "error": (
                f"`{column}` is not available in the selected MCP data. "
                "This tool will not synthesize soil chemistry from spectral bands. "
                "Provide a measured CSV containing this column to plot it."
            ),
        }

    pyplot = _apply_global_plot_styles()
    fig, ax = pyplot.subplots(figsize=(6, 4.5))
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return {
            "success": False,
            "status": "insufficient_data",
            "plot_name": plot_name,
            "path": tif_path,
            "missing_fields": [column],
            "available_fields": list(df.columns),
            "metadata": metadata,
            "error": f"`{column}` is present but has no numeric values to plot.",
        }
    mean_value = float(values.mean())
    ax.hist(values, bins=30, color=color, edgecolor="black", alpha=0.72)
    ax.axvline(mean_value, color="darkred", linestyle="--", label=mean_label_format.format(mean_value))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    result = _save_plot(fig, tif_path, plot_name, output_path, image_format)
    result.update({
        "source": "mcp_tools_only",
        "metric": column,
        "mean": _clean_float(mean_value, 4),
        "metadata": metadata,
    })
    return result


def _soil_histogram_tool(
    plot_name: str,
    tif_path: str,
    output_path: str,
    image_format: str,
    max_pixels: int,
    color: str | None = None,
) -> dict[str, Any]:
    spec = SOIL_PLOT_SPECS[plot_name]
    try:
        return _histogram_plot(
            tif_path=tif_path,
            output_path=output_path,
            image_format=image_format,
            max_pixels=max_pixels,
            column=spec["column"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            color=color or spec["default_color"],
            mean_label_format=spec["mean_label_format"],
            plot_name=plot_name,
        )
    except Exception as exc:
        return _tool_error(plot_name, tif_path, exc)


def _plot_markdown(plot_result: dict[str, Any]) -> str | None:
    path = plot_result.get("relative_path")
    if not path:
        return None
    title = str(plot_result.get("plot_name") or Path(path).stem).replace("_", " ").title()
    url = f"/api/data?path={path}"
    chart_type = str(plot_result.get("chart_type") or Path(path).suffix.lstrip(".")).lower()
    if chart_type in {"svg", "png"}:
        return f"![{title}]({url})"
    return f"[{title}]({url})"


def _collect_successful_plots(plot_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    for payload in plot_payloads:
        if payload.get("success") and payload.get("relative_path"):
            plots.append(payload)
        for nested in payload.get("plots") or []:
            if nested.get("success") and nested.get("relative_path"):
                plots.append(nested)
    return plots


def _collect_failed_plots(plot_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for payload in plot_payloads:
        if payload.get("success") is False:
            failures.append({
                "plot_name": payload.get("plot_name") or payload.get("tool") or "unknown_plot",
                "error": payload.get("error") or payload.get("warning") or "Plot generation failed.",
                "missing_fields": payload.get("missing_fields") or [],
            })
        for nested in payload.get("plots") or []:
            if nested.get("success") is False:
                failures.append({
                    "plot_name": nested.get("plot_name") or "unknown_plot",
                    "error": nested.get("error") or nested.get("warning") or "Plot generation failed.",
                    "missing_fields": nested.get("missing_fields") or [],
                })
    return failures


def _visualization_response(
    *,
    parcel: dict[str, Any] | None = None,
    visualizations: list[dict[str, Any]] | None = None,
    tools_used: list[str] | None = None,
    status: str = "insufficient",
    missing_fields: list[str] | None = None,
    warnings: list[str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    methodology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "parcel_visualization",
        "source": "mcp_tools_only",
        "parcel": parcel,
        "visualizations": visualizations or [],
        "metadata": {
            "tools_used": tools_used or [],
            "data_validation": {
                "status": status,
                "missing_fields": missing_fields or [],
            },
            "methodology": methodology or {},
            "warnings": warnings or [],
            "errors": errors or [],
        },
    }


def _selected_parcel_path(tif_path: str = "", paths: list[str] | None = None) -> tuple[str | None, list[str]]:
    requested_paths = [path for path in (paths or []) if str(path).strip()]
    if tif_path and tif_path.strip():
        return tif_path.strip(), []
    if len(requested_paths) == 1:
        return requested_paths[0], []
    if len(requested_paths) > 1:
        return None, requested_paths

    discovered = _discover_tif_paths()
    if len(discovered) == 1:
        return discovered[0], []
    return None, discovered


def _selected_parcel_paths(tif_path: str = "", paths: list[str] | None = None) -> list[str]:
    if tif_path and tif_path.strip():
        return [tif_path.strip()]
    requested_paths = [path for path in (paths or []) if str(path).strip()]
    if requested_paths:
        return requested_paths
    return _discover_tif_paths()


@logged_mcp_tool
def list_data_files(subdir: str = ".", pattern: str = "*", max_results: int = 100) -> list[dict[str, Any]]:
    """List files and folders inside the mounted shared data directory."""
    root = _safe_path(subdir)
    if not root.exists():
        return [{"error": f"Subdir does not exist: {subdir}"}]
    if not root.is_dir():
        return [{"error": f"Not a directory: {subdir}"}]

    entries = []
    for path in sorted(root.glob(pattern)):
        try:
            entries.append(_file_info(path))
        except Exception as exc:
            entries.append({"path": str(path), "error": str(exc)})
        if len(entries) >= max_results:
            break
    return entries


@logged_mcp_tool
def read_data_file(path: str, max_chars: int = 20000) -> str:
    """Read a text-like file from the shared data directory."""
    file_path = _safe_path(path)
    if not file_path.exists():
        return f"File does not exist: {path}"
    if not file_path.is_file():
        return f"Not a file: {path}"

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read(max_chars + 1)

    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[TRUNCATED after {max_chars} characters]"
    return content


@logged_mcp_tool
def write_data_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write a text file into DATA_ROOT. Disabled unless ALLOW_WRITE_TO_DATA=true."""
    if not ALLOW_WRITE_TO_DATA:
        return "Writing is disabled. Set ALLOW_WRITE_TO_DATA=true to enable this tool."

    file_path = _safe_path(path)
    if file_path.exists() and not overwrite:
        return f"File already exists and overwrite=false: {path}"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"

@logged_mcp_tool
def search_data_files(query: str, subdir: str = ".", glob_pattern: str = "**/*", max_matches: int = 20) -> list[dict[str, Any]]:
    """Search for text inside files in the shared data directory."""
    root = _safe_path(subdir)
    if not root.exists() or not root.is_dir():
        return [{"error": f"Invalid directory: {subdir}"}]

    query_lower = query.lower()
    matches: list[dict[str, Any]] = []

    for file_path in sorted(root.glob(glob_pattern)):
        if not file_path.is_file():
            continue

        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if query_lower in line.lower():
                        matches.append({
                            "relative_path": str(file_path.relative_to(DATA_ROOT)),
                            "line": line_no,
                            "text": line.strip()[:500],
                        })
                        if len(matches) >= max_matches:
                            return matches
        except Exception:
            continue

    return matches


@logged_mcp_tool
def get_data_root_info() -> dict[str, Any]:
    """Return basic information about the mounted shared data directory."""
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "data_root": str(DATA_ROOT),
            "configured_data_root": str(CONFIGURED_DATA_ROOT),
            "exists": DATA_ROOT.exists(),
            "allow_write": ALLOW_WRITE_TO_DATA,
            "error": f"Could not create or access DATA_ROOT: {exc}",
            "examples": [],
        }

    entries = list(DATA_ROOT.iterdir())
    return {
        "data_root": str(DATA_ROOT),
        "configured_data_root": str(CONFIGURED_DATA_ROOT),
        "exists": DATA_ROOT.exists(),
        "allow_write": ALLOW_WRITE_TO_DATA,
        "top_level_items": len(entries),
        "examples": [_file_info(p) for p in sorted(entries)[:20]],
    }


@logged_mcp_tool
def inspect_geotiff(path: str) -> dict[str, Any]:
    """Inspect GeoTIFF dimensions, CRS, bounds, and center coordinates."""
    try:
        rasterio = _rasterio()
        tif_path = _resolve_tif_path(path)
        if not tif_path.exists():
            return {"success": False, "path": path, "error": f"GeoTIFF file does not exist: {path}"}

        with rasterio.open(tif_path) as src:
            return {"success": True, **_geotiff_metadata(src, tif_path)}
    except Exception as exc:
        return {"success": False, "path": path, "error": f"{type(exc).__name__}: {exc}"}


@logged_mcp_tool
def analyze_enmap_parcel(tif_path: str) -> dict[str, Any]:
    """Calculate verified vegetation, coverage, moisture, and ranking metrics for one EnMap GeoTIFF."""
    try:
        return {"success": True, **_analyze_tif(tif_path)}
    except Exception as exc:
        return {"success": False, "path": tif_path, "error": f"{type(exc).__name__}: {exc}"}


@logged_mcp_tool
def compare_enmap_parcels(paths: list[str] = []) -> dict[str, Any]:
    """Analyze and rank multiple EnMap GeoTIFF parcels using transparent NDVI-derived metrics."""
    if not paths:
        paths = _discover_tif_paths()

    analyses = []
    errors = []
    for path in paths:
        result = analyze_enmap_parcel(path)
        if result.get("success"):
            analyses.append(result)
        else:
            errors.append(result)

    ranked = sorted(
        analyses,
        key=lambda item: item.get("metrics", {}).get("investment_score")
        if item.get("metrics", {}).get("investment_score") is not None
        else -9999,
        reverse=True,
    )

    comparison = []
    for rank, item in enumerate(ranked, start=1):
        metrics = item["metrics"]
        comparison.append({
            "rank": rank,
            "parcel": item["name"],
            "path": item["path"],
            "mean_ndvi": metrics["mean_ndvi"],
            "ndvi_std": metrics["ndvi_std"],
            "healthy_coverage_pct": metrics["healthy_coverage_pct"],
            "stressed_coverage_pct": metrics["stressed_coverage_pct"],
            "investment_score": metrics["investment_score"],
            "valid_pixels": metrics["valid_pixels"],
        })

    return {
        "success": bool(analyses),
        "best_parcel": comparison[0] if comparison else None,
        "comparison": comparison,
        "analyses": analyses,
        "errors": errors,
        "methodology": {
            "ranking": "Parcels are sorted descending by investment_score.",
            "bands": {
                "red": "Band 40",
                "nir": "Band 80",
                "swir": "Band 200 when available",
            },
        },
    }


@logged_mcp_tool
def create_parcel_metric_chart(paths: list[str] = [], output_path: str = "plots/parcel_metric_comparison.svg") -> dict[str, Any]:
    """Create an SVG bar chart for parcel NDVI, healthy coverage, and investment score."""
    try:
        pyplot = _apply_global_plot_styles()
    except Exception as exc:
        return {"success": False, "error": f"Matplotlib is unavailable: {exc}"}

    comparison = compare_enmap_parcels(paths)
    rows = comparison.get("comparison") or []
    if not rows:
        return {"success": False, "error": "No parcel metrics available to plot.", "details": comparison}

    labels = [row["parcel"].replace("_mosaic.TIF", "") for row in rows]
    ndvi = [row["mean_ndvi"] or 0 for row in rows]
    coverage = [(row["healthy_coverage_pct"] or 0) / 100.0 for row in rows]
    score = [row["investment_score"] or 0 for row in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = pyplot.subplots(figsize=(11, 5.5))
    ax.bar(x - width, ndvi, width, label="Mean NDVI")
    ax.bar(x, coverage, width, label="Healthy coverage fraction")
    ax.bar(x + width, np.array(score) / 100.0, width, label="Score / 100")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Parcel Metric Comparison")
    ax.set_ylabel("Normalized value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_file = _safe_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="svg", bbox_inches="tight")
    pyplot.close(fig)

    return {
        "success": True,
        "relative_path": str(output_file.relative_to(DATA_ROOT.resolve())),
        "chart_type": "svg",
        "plotted_rows": rows,
    }


@logged_mcp_tool
def plot_ph_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#4f81bd",
) -> dict[str, Any]:
    """Create a pH distribution plot only when measured pH data is present."""
    return _soil_histogram_tool("ph_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_nitrogen_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#c0504d",
) -> dict[str, Any]:
    """Create a nitrogen distribution plot only when measured nitrogen data is present."""
    return _soil_histogram_tool("nitrogen_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_phosphorus_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#9bbb59",
) -> dict[str, Any]:
    """Create a phosphorus distribution plot only when measured phosphorus data is present."""
    return _soil_histogram_tool("phosphorus_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_potassium_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#8064a2",
) -> dict[str, Any]:
    """Create a potassium distribution plot only when measured potassium data is present."""
    return _soil_histogram_tool("potassium_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_magnesium_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#4bacc6",
) -> dict[str, Any]:
    """Create a magnesium distribution plot only when measured magnesium data is present."""
    return _soil_histogram_tool("magnesium_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_som_profile(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
    color: str = "#f79646",
) -> dict[str, Any]:
    """Create a soil organic matter distribution plot only when measured SOM data is present."""
    return _soil_histogram_tool("som_profile", tif_path, output_path, image_format, max_pixels, color)


@logged_mcp_tool
def plot_ndvi_vs_swi_scatter(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
) -> dict[str, Any]:
    """Create an NDVI vs SWI scatter plot from directly computed EnMap spectral indices."""
    try:
        _, df, metadata = _process_enmap_soil_data(tif_path, max_pixels=max_pixels)
        if "NDVI" not in df.columns or "SWI" not in df.columns:
            return {
                "success": False,
                "status": "insufficient_data",
                "plot_name": "ndvi_vs_swi_scatter",
                "path": tif_path,
                "missing_fields": ["NDVI", "SWI"],
                "available_fields": list(df.columns),
                "metadata": metadata,
                "error": "NDVI and SWI are required and were not available in the selected MCP data.",
            }
        pyplot = _apply_global_plot_styles()
        fig, ax = pyplot.subplots(figsize=(7, 5.5))
        scatter = ax.scatter(
            df["NDVI"],
            df["SWI"],
            c=df["SWI"],
            cmap="viridis",
            alpha=0.6,
            s=25,
        )
        ax.set_title("NDVI vs SWI", fontweight="bold")
        ax.set_xlabel("NDVI (Vegetation Index)")
        ax.set_ylabel("SWI (Soil Water Index)")
        cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("SWI", size=9)
        result = _save_plot(fig, tif_path, "ndvi_vs_swi_scatter", output_path, image_format)
        result.update({
            "source": "mcp_tools_only",
            "x": "NDVI",
            "y": "SWI",
            "color": "SWI",
            "metadata": metadata,
        })
        return result
    except Exception as exc:
        return {"success": False, "plot_name": "ndvi_vs_swi_scatter", "path": tif_path, "error": f"{type(exc).__name__}: {exc}"}


@logged_mcp_tool
def render_agromanagement_textbox(
    tif_path: str,
    output_path: str = "",
    image_format: str = "svg",
    max_pixels: int = 250000,
) -> dict[str, Any]:
    """Create a rendered textbox from verified measured fields or spectral indices."""
    try:
        report, df, metadata = _process_enmap_soil_data(tif_path, max_pixels=max_pixels)
        pyplot = _apply_global_plot_styles()
        fig, ax = pyplot.subplots(figsize=(6.5, 4.8))
        ax.axis("off")
        rows = ["=== VERIFIED MCP DATA ===", "", f"Total Samples Bound : {len(df)} rows/pixels"]
        for field, label, value_format in TEXTBOX_FIELDS:
            if field in df.columns:
                rows.append(f"{label:<22}: {value_format.format(float(df[field].mean()))}")
        if metadata.get("unavailable_without_measured_table"):
            rows.extend([
                "",
                "Unavailable fields were not synthesized.",
                "Provide measured CSV columns to plot soil chemistry.",
            ])
        summary_text = "\n".join(rows)
        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox={"boxstyle": "round,pad=1", "facecolor": "#f8f9fa", "edgecolor": "#cccccc"},
        )
        result = _save_plot(fig, tif_path, "agromanagement_textbox", output_path, image_format)
        result.update({
            "source": "mcp_tools_only",
            "report": report,
            "metadata": metadata,
        })
        return result
    except Exception as exc:
        return {"success": False, "plot_name": "agromanagement_textbox", "path": tif_path, "error": f"{type(exc).__name__}: {exc}"}


@logged_mcp_tool
def create_agromanagement_plot_suite(
    tif_path: str,
    output_dir: str = "plots",
    image_format: str = "svg",
    max_pixels: int = 250000,
) -> dict[str, Any]:
    """Create all agromanagement plots for one EnMap GeoTIFF and return their saved chart paths."""
    results = []
    for plot_name in SOIL_PLOT_SPECS:
        output_path = str(Path(output_dir) / Path(_default_plot_path(tif_path, plot_name, image_format)).name)
        results.append(_soil_histogram_tool(
            plot_name,
            tif_path,
            output_path,
            image_format,
            max_pixels,
        ))

    for plot_name, plot_func in (
        ("ndvi_vs_swi_scatter", plot_ndvi_vs_swi_scatter),
        ("agromanagement_textbox", render_agromanagement_textbox),
    ):
        output_path = str(Path(output_dir) / Path(_default_plot_path(tif_path, plot_name, image_format)).name)
        results.append(plot_func(
            tif_path=tif_path,
            output_path=output_path,
            image_format=image_format,
            max_pixels=max_pixels,
        ))

    successful = [result for result in results if result.get("success")]
    failed = [result for result in results if not result.get("success")]
    return {
        "success": bool(successful),
        "status": "valid" if len(successful) == len(results) else "partial" if successful else "insufficient_data",
        "source": "mcp_tools_only",
        "tif_path": tif_path,
        "warning": (
            "Soil chemistry plots are skipped unless measured table columns are present. "
            "No soil chemistry values are synthesized from GeoTIFF bands."
        ),
        "plots": results,
        "successful_plots": len(successful),
        "failed_plots": len(failed),
    }


@logged_mcp_tool
def build_parcel_visualization(tif_path: str = "", paths: list[str] = []) -> dict[str, Any]:
    """Build frontend-ready map, chart, and table specs for one selected EnMap parcel."""
    selected_path, ambiguous_paths = _selected_parcel_path(tif_path, paths)
    if not selected_path:
        return _visualization_response(
            tools_used=["build_parcel_visualization"],
            status="insufficient",
            missing_fields=["selected_parcel"],
            warnings=[
                "No single selected parcel was provided. Select exactly one GeoTIFF parcel before requesting visualization."
            ],
            errors=[{
                "code": "selected_parcel_required",
                "available_paths": ambiguous_paths,
            }],
        )

    tools_used = ["build_parcel_visualization", "inspect_geotiff", "analyze_enmap_parcel"]
    warnings: list[str] = []
    errors: list[dict[str, Any]] = []

    try:
        geo = inspect_geotiff(selected_path)
    except Exception as exc:
        geo = {"success": False, "path": selected_path, "error": f"{type(exc).__name__}: {exc}"}

    try:
        analysis = analyze_enmap_parcel(selected_path)
    except Exception as exc:
        analysis = {"success": False, "path": selected_path, "error": f"{type(exc).__name__}: {exc}"}

    if not geo.get("success"):
        errors.append({"tool": "inspect_geotiff", "error": geo.get("error", "GeoTIFF inspection failed.")})
    if not analysis.get("success"):
        errors.append({"tool": "analyze_enmap_parcel", "error": analysis.get("error", "Parcel metric analysis failed.")})

    if errors:
        return _visualization_response(
            tools_used=tools_used,
            status="insufficient",
            missing_fields=["geometry", "metrics"],
            warnings=["Visualization cannot be produced because required MCP tool output is missing."],
            errors=errors,
        )

    bounds = geo.get("bounds") or {}
    center = geo.get("center") or {}
    metrics = analysis.get("metrics") or {}
    missing_fields = [
        field
        for field in (
            "bounds",
            "center",
            "mean_ndvi",
            "ndvi_std",
            "healthy_coverage_pct",
            "stressed_coverage_pct",
            "investment_score",
        )
        if (
            (field == "bounds" and not bounds)
            or (field == "center" and not center)
            or (field not in {"bounds", "center"} and metrics.get(field) is None)
        )
    ]

    parcel = {
        "id": analysis.get("path") or geo.get("path") or selected_path,
        "geometry": {
            "type": "bbox",
            "bounds": bounds,
            "center": center,
            "crs": geo.get("crs"),
        },
        "properties": {
            "name": analysis.get("name") or geo.get("name"),
            "path": analysis.get("path") or geo.get("path"),
            "width": geo.get("width"),
            "height": geo.get("height"),
            "bands": geo.get("bands"),
        },
    }

    visualizations: list[dict[str, Any]] = []
    if bounds and center:
        visualizations.append({
            "type": "map",
            "title": "Parcel extent",
            "data": [{
                "parcel_id": parcel["id"],
                "bounds": bounds,
                "center": center,
                "crs": geo.get("crs"),
            }],
            "encoding": {
                "geometry": "bounds",
                "label": "parcel_id",
                "center": "center",
            },
            "units": {
                "coordinates": geo.get("crs") or "unknown",
            },
            "rendering_notes": "Render the parcel as a bounding box using the supplied CRS and center coordinates. Do not infer parcel boundaries beyond this bbox.",
        })

    metric_rows = []
    metric_specs = [
        ("mean_ndvi", "Mean NDVI", "index"),
        ("ndvi_std", "NDVI standard deviation", "index"),
        ("healthy_coverage_pct", "Healthy coverage", "percent"),
        ("stressed_coverage_pct", "Stressed coverage", "percent"),
        ("investment_score", "Investment score", "score"),
    ]
    for key, label, unit in metric_specs:
        value = metrics.get(key)
        if value is not None:
            metric_rows.append({
                "parcel_id": parcel["id"],
                "metric": label,
                "field": key,
                "value": value,
                "unit": unit,
            })

    if metric_rows:
        visualizations.append({
            "type": "bar_chart",
            "title": "Verified parcel metrics",
            "data": metric_rows,
            "encoding": {
                "x": "metric",
                "y": "value",
                "color": "metric",
                "label": "value",
            },
            "units": {
                "mean_ndvi": "index",
                "ndvi_std": "index",
                "healthy_coverage_pct": "percent",
                "stressed_coverage_pct": "percent",
                "investment_score": "score",
            },
            "rendering_notes": "Render only the supplied rows. Values are already calculated by MCP tools and must not be normalized unless the UI clearly labels that transform.",
        })

    visualizations.append({
        "type": "table",
        "title": "Verified parcel data",
        "data": [{
            "parcel_id": parcel["id"],
            "parcel_name": parcel["properties"]["name"],
            **metrics,
        }],
        "encoding": {
            "columns": list(metrics.keys()),
            "label": "parcel_name",
        },
        "units": {
            "mean_ndvi": "index",
            "ndvi_std": "index",
            "healthy_coverage_pct": "percent",
            "stressed_coverage_pct": "percent",
            "investment_score": "score",
        },
        "rendering_notes": "Display values exactly as returned. Blank cells should mean null or unavailable data, not zero.",
    })

    if missing_fields:
        warnings.append("Some optional chart fields were unavailable; charts that require them were omitted or reduced.")

    return _visualization_response(
        parcel=parcel,
        visualizations=visualizations,
        tools_used=tools_used,
        status="partial" if missing_fields else "valid",
        missing_fields=missing_fields,
        warnings=warnings,
        methodology=analysis.get("methodology") or {},
    )


@logged_mcp_tool
def run_full_geo_report_flow(
    tif_path: str = "",
    paths: list[str] = [],
    output_dir: str = REPORTS_DIR,
    image_format: str = "svg",
    max_pixels: int = 250000,
) -> dict[str, Any]:
    """Run the full report flow: read data, create geo/risk reports, save data files, generate plots, and write markdown."""
    selected_paths = _selected_parcel_paths(tif_path, paths)
    if not selected_paths:
        return {
            "success": False,
            "status": "insufficient_data",
            "error": "No selected GeoTIFF parcels were provided and no GeoTIFF files were found in DATA_ROOT.",
            "data_root": str(DATA_ROOT),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_slug = _slug(Path(selected_paths[0]).stem if len(selected_paths) == 1 else "parcel-comparison")
    base_name = f"{timestamp}-{report_slug}"
    report_dir = f"{output_dir}/{base_name}"
    plot_output_dir = "plots"
    tool_outputs: dict[str, Any] = {
        "selected_paths": selected_paths,
        "data_inventory": list_data_files(max_results=500),
        "read_files": [],
        "parcel_reports": [],
        "comparison": None,
        "comparison_chart": None,
        "plots": [],
        "plot_failures": [],
        "plot_directory": {
            "relative_path": "plots",
            "writable": None,
            "error": None,
        },
    }

    try:
        plots_dir = _safe_path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        probe = plots_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        tool_outputs["plot_directory"]["writable"] = True
    except Exception as exc:
        tool_outputs["plot_directory"]["writable"] = False
        tool_outputs["plot_directory"]["error"] = f"{type(exc).__name__}: {exc}"
        plot_output_dir = f"{report_dir}/plots"

    for item in tool_outputs["data_inventory"]:
        relative_path = item.get("relative_path")
        if not relative_path or item.get("type") != "file":
            continue
        suffix = Path(relative_path).suffix.lower()
        if suffix in {".csv", ".txt", ".md", ".json", ".geojson"}:
            tool_outputs["read_files"].append({
                "path": relative_path,
                "content": read_data_file(relative_path, max_chars=20000),
            })

    for parcel_path in selected_paths:
        geo_report = create_geo_and_risk_report(parcel_path)
        geo = inspect_geotiff(parcel_path)
        analysis = analyze_enmap_parcel(parcel_path)
        visualization = build_parcel_visualization(tif_path=parcel_path)
        plot_suite = create_agromanagement_plot_suite(
            tif_path=parcel_path,
            output_dir=plot_output_dir,
            image_format=image_format,
            max_pixels=max_pixels,
        )
        tool_outputs["parcel_reports"].append({
            "path": parcel_path,
            "geo_and_risk_report": geo_report,
            "geotiff": geo,
            "analysis": analysis,
            "visualization": visualization,
            "plot_suite": plot_suite,
        })
        tool_outputs["plots"].extend(_collect_successful_plots([plot_suite]))
        tool_outputs["plot_failures"].extend(_collect_failed_plots([plot_suite]))

    comparison = compare_enmap_parcels(selected_paths)
    comparison_chart = create_parcel_metric_chart(
        selected_paths,
        output_path=f"{plot_output_dir}/{base_name}-parcel-metric-comparison.{image_format}",
    )
    tool_outputs["comparison"] = comparison
    tool_outputs["comparison_chart"] = comparison_chart
    tool_outputs["plots"].extend(_collect_successful_plots([comparison_chart]))
    tool_outputs["plot_failures"].extend(_collect_failed_plots([comparison_chart]))

    data_file = _write_data_json(f"{report_dir}/tool_outputs.json", tool_outputs)

    markdown_lines = [
        "# Parcel Geo And Risk Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Source Data Read",
        "",
        f"- Data root: `{DATA_ROOT}`",
        f"- Selected parcels: {', '.join(f'`{path}`' for path in selected_paths)}",
        f"- Inventory entries: {len(tool_outputs['data_inventory'])}",
        f"- Text/table files read: {len(tool_outputs['read_files'])}",
        f"- Full tool-output data file: [`{data_file['relative_path']}`](/api/data?path={data_file['relative_path']})",
        "",
    ]

    if tool_outputs["comparison"]:
        comparison_rows = tool_outputs["comparison"].get("comparison") or []
        markdown_lines.extend([
            "## Verified Parcel Comparison",
            "",
            "| Rank | Parcel | Mean NDVI | NDVI Std | Healthy Coverage % | Stressed Coverage % | Score |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        for row in comparison_rows:
            markdown_lines.append(
                f"| {row.get('rank')} | {row.get('parcel')} | {row.get('mean_ndvi')} | "
                f"{row.get('ndvi_std')} | {row.get('healthy_coverage_pct')} | "
                f"{row.get('stressed_coverage_pct')} | {row.get('investment_score')} |"
            )
        markdown_lines.append("")

    for parcel in tool_outputs["parcel_reports"]:
        analysis = parcel.get("analysis") or {}
        metrics = analysis.get("metrics") or {}
        markdown_lines.extend([
            f"## Parcel: {parcel.get('path')}",
            "",
            "### Geo And Risk Report",
            "",
            "```text",
            str(parcel.get("geo_and_risk_report") or ""),
            "```",
            "",
        ])
        if metrics:
            markdown_lines.extend([
                "### Verified Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ])
            for key, value in metrics.items():
                markdown_lines.append(f"| {key} | {value} |")
            markdown_lines.append("")

    plot_markdown = [_plot_markdown(plot) for plot in tool_outputs["plots"]]
    plot_markdown = [line for line in plot_markdown if line]
    markdown_lines.extend(["## Figures", ""])
    if plot_markdown:
        markdown_lines.extend(plot_markdown)
        markdown_lines.append("")
    else:
        markdown_lines.extend([
            "No figures were generated.",
            "",
        ])

    if tool_outputs["plot_directory"].get("writable") is False:
        markdown_lines.extend([
            "### Figure Output Directory Error",
            "",
            f"- `data/plots` is not writable: {tool_outputs['plot_directory'].get('error')}",
            "",
        ])

    if tool_outputs["plot_failures"]:
        markdown_lines.extend([
            "### Figure Generation Failures",
            "",
            "| Plot | Missing Fields | Error |",
            "|---|---|---|",
        ])
        for failure in tool_outputs["plot_failures"]:
            missing = ", ".join(str(item) for item in failure.get("missing_fields") or []) or "-"
            error = str(failure.get("error") or "Plot generation failed.").replace("|", "\\|")
            markdown_lines.append(f"| {failure.get('plot_name')} | {missing} | {error} |")
        markdown_lines.append("")

    markdown_lines.extend([
        "## Unverified Data / Risks",
        "",
        "- Flood, road, and external terrain risk are only reported when the corresponding MCP tool has verified supporting data.",
        "- Soil chemistry plots are only generated from measured table columns. They are not synthesized from GeoTIFF bands.",
        "",
    ])

    report_markdown = "\n".join(markdown_lines)
    report_file = _write_data_markdown(f"{report_dir}/report.md", report_markdown)

    return {
        "success": True,
        "status": "valid" if tool_outputs["plots"] else "partial",
        "source": "mcp_tools_only",
        "selected_paths": selected_paths,
        "report_markdown": report_markdown,
        "report_file": report_file,
        "data_file": data_file,
        "plots": tool_outputs["plots"],
        "tool_outputs": tool_outputs,
    }


@logged_mcp_tool
def create_geo_and_risk_report(tif_path: str) -> str:
    """Generate a compact geospatial, accessibility, and EnMap metric report for a parcel."""
    geo = inspect_geotiff(tif_path)
    metrics = analyze_enmap_parcel(tif_path)

    lines = ["[GEO AND PARCEL RISK REPORT]"]
    if geo.get("success"):
        center = geo.get("center") or {}
        lines.extend([
            f"Path: {geo.get('path')}",
            f"CRS: {geo.get('crs')}",
            f"Center: lat={center.get('lat')}, lon={center.get('lon')}",
            f"Raster: {geo.get('width')} x {geo.get('height')} pixels, {geo.get('bands')} bands",
        ])
    else:
        lines.append(f"GeoTIFF inspection failed: {geo.get('error')}")

    if metrics.get("success"):
        values = metrics["metrics"]
        lines.extend([
            "[VERIFIED ENMAP METRICS]",
            f"Mean NDVI: {values.get('mean_ndvi')}",
            f"NDVI Standard Deviation: {values.get('ndvi_std')}",
            f"Healthy Coverage Percentage: {values.get('healthy_coverage_pct')}",
            f"Stressed Coverage Percentage: {values.get('stressed_coverage_pct')}",
            f"Investment Score: {values.get('investment_score')}",
            "Flood, road, and external terrain risk are not computed by this tool unless supporting DEM/OSM dependencies and data are explicitly available.",
        ])
    else:
        lines.append(f"Metric analysis failed: {metrics.get('error')}")

    return "\n".join(lines)


if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="streamable-http")
