from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import rasterio
from bmi_topography import Topography
import rioxarray
import shapely
from mcp.server.fastmcp import FastMCP
from pyproj import Transformer
from pysheds.grid import Grid
from rasterio.crs import CRS
from shapely.geometry import Point
import seaborn as sns
from src.common_config import DATA_ROOT, env_bool, env_int, env_str

MCP_HOST = env_str("MCP_HOST", "0.0.0.0")
MCP_PORT = env_int("MCP_PORT", 8000)
ALLOW_WRITE_TO_DATA = env_bool("ALLOW_WRITE_TO_DATA", False)

mcp = FastMCP(
    "local-data-toolset",
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path="/mcp",
)


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
        raise ValueError(
            f"Path is outside DATA_ROOT. Use a path under {DATA_ROOT} or a path relative to it."
        )

    return _safe_path(raw_path)


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "relative_path": str(path.relative_to(DATA_ROOT)),
        "type": "dir" if path.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified_epoch": int(stat.st_mtime),
    }
def _tif_to_dataframe(tif_path: str) -> pd.DataFrame:
    "THIS FUNCTION CONVERTS A HYPERSPECTRAL TIFF INTO A FLAT DATAFRAME WITH AUTO-SCALE AND INTERPOLATION"
    print(f"Opening geospatial raster: {tif_path}")
    
    with rasterio.open(tif_path) as src:
        num_bands = src.count
        height = src.height
        width = src.width
        num_pixels = height * width
        
        print(f"Image Dimensions: {height}x{width} ({num_pixels} total pixels)")
        print(f"Detected Spectral Layers: {num_bands} bands")
        
        img_data = src.read()       
        indices = np.arange(num_pixels)
            
        band_data = {}
        for b in range(1, num_bands + 1):
            band_flattened = img_data[b-1].flatten()
            sampled_pixels = band_flattened[indices]
            
            if np.nanmax(sampled_pixels) > 1.0:
                sampled_pixels = sampled_pixels / 10000.0
                
            band_data[f'Band_{b}'] = sampled_pixels
            
        df = pd.DataFrame(band_data)
        df.insert(0, 'Pixel_ID', np.arange(len(df)))
        
        print("Dataframe conversion complete.")
        return df


@mcp.tool()
def list_data_files(subdir: str = ".", pattern: str = "*", max_results: int = 100) -> list[dict[str, Any]]:
    """List files and folders inside the mounted shared data directory.

    Args:
        subdir: Subfolder under DATA_ROOT. Use "." for the root.
        pattern: Glob pattern, for example "*.csv" or "**/*.txt".
        max_results: Maximum number of entries to return.
    """
    root = _safe_path(subdir)
    if not root.exists():
        return [{"error": f"Subdir does not exist: {subdir}"}]
    if not root.is_dir():
        return [{"error": f"Not a directory: {subdir}"}]

    entries = []
    for path in sorted(root.glob(pattern)):
        try:
            entries.append(_file_info(path))
        except Exception as exc:  # Defensive: do not break the whole tool call.
            entries.append({"path": str(path), "error": str(exc)})
        if len(entries) >= max_results:
            break
    return entries


@mcp.tool()
def read_text_file(path: str, max_chars: int = 20000) -> str:
    """Read a text-like file from the shared data directory.

    Args:
        path: Relative file path under DATA_ROOT.
        max_chars: Maximum characters to return.
    """
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


@mcp.tool()
def write_text_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write a text file into the shared data directory. Disabled unless ALLOW_WRITE_TO_DATA=true.

    Args:
        path: Relative file path under DATA_ROOT.
        content: Text content to write.
        overwrite: Whether to overwrite an existing file.
    """
    if not ALLOW_WRITE_TO_DATA:
        return "Writing is disabled. Set ALLOW_WRITE_TO_DATA=true to enable this tool."

    file_path = _safe_path(path)
    if file_path.exists() and not overwrite:
        return f"File already exists and overwrite=false: {path}"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


@mcp.tool()
def summarize_csv(path: str, max_rows: int = 5) -> dict[str, Any]:
    """Summarize a CSV file in the shared data directory.

    Args:
        path: Relative CSV path under DATA_ROOT.
        max_rows: Number of preview rows to return.
    """
    file_path = _safe_path(path)
    if not file_path.exists():
        return {"error": f"File does not exist: {path}"}
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}

    df = pd.read_csv(file_path)
    return {
        "path": path,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "preview": df.head(max_rows).to_dict(orient="records"),
        "numeric_summary": df.describe(include="number").fillna("").to_dict(),
    }


@mcp.tool()
def search_text_files(query: str, subdir: str = ".", glob_pattern: str = "**/*", max_matches: int = 20) -> list[dict[str, Any]]:
    """Search for text inside files in the shared data directory.

    Args:
        query: Case-insensitive text to search for.
        subdir: Subfolder under DATA_ROOT.
        glob_pattern: File glob pattern, for example "**/*.md".
        max_matches: Maximum matching lines to return.
    """
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
            # Binary or unreadable file; skip.
            continue

    return matches


@mcp.tool()
def get_data_root_info() -> dict[str, Any]:
    """Return basic information about the mounted shared data directory."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    entries = list(DATA_ROOT.iterdir())
    return {
        "data_root": str(DATA_ROOT),
        "exists": DATA_ROOT.exists(),
        "allow_write": ALLOW_WRITE_TO_DATA,
        "top_level_items": len(entries),
        "examples": [_file_info(p) for p in sorted(entries)[:10]],
    }

@mcp.tool()
def _extract_tif_midpoint(tif_path: str) -> dict[str, Any]:
    """Extracts bounds and center of a TIFF, automatically converting 
    projected meter coordinates (like Greek EPSG:2100) back to standard Lat/Lon.
    """
    try:
        file_path = _resolve_data_path(tif_path)
        if not file_path.exists():
            return {"success": False, "error": f"GeoTIFF file does not exist: {file_path}"}
        if not file_path.is_file():
            return {"success": False, "error": f"GeoTIFF path is not a file: {file_path}"}

        with rasterio.open(file_path) as src:
            bounds = src.bounds
            bbox_native = {
                "west": bounds.left, "east": bounds.right,
                "south": bounds.bottom, "north": bounds.top
            }
            
            mid_native_x = (bbox_native["west"] + bbox_native["east"]) / 2.0
            mid_native_y = (bbox_native["south"] + bbox_native["north"]) / 2.0
            
            native_crs = src.crs
            
            if native_crs and not native_crs.is_geographic:
                print(f"Detected projected native CRS: {native_crs.to_string()}")
                transformer = Transformer.from_crs(native_crs, "EPSG:4326", always_xy=True)
                mid_lon, mid_lat = transformer.transform(mid_native_x, mid_native_y)
            else:
                mid_lon, mid_lat = mid_native_x, mid_native_y
            
            return {
                "center_coords": {"lat": mid_lat, "lon": mid_lon},
                "center_pixel_index": {"row": src.height // 2, "col": src.width // 2},
                "crs": native_crs.to_string() if native_crs else "Unknown",
                "success": True
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _bbox_from_point(target_lat: float, target_lon: float, distance_km: float = 2.0) -> dict[str, float]:
    lat_offset = (distance_km / 111.0)
    lon_offset = (distance_km / (111.0 * np.cos(np.radians(target_lat))))

    return {
        "south": target_lat - lat_offset,
        "north": target_lat + lat_offset,
        "west": target_lon - lon_offset,
        "east": target_lon + lon_offset
    }


def _Flood_risk(target_lat: float, target_lon: float) -> str:
    bbox = _bbox_from_point(target_lat, target_lon)
    print("Fetching live Copernicus 30m DEM from OpenTopography API...")

    topo = Topography(
        dem_type="COP30",
        output_format="GTiff",
        south=bbox["south"],
        north=bbox["north"],
        west=bbox["west"],
        east=bbox["east"]
    )

    da = topo.load()
    tif_filename = "temp_greece_dem.tif"
    da.rio.to_raster(tif_filename)

    print("DEM Download Complete. Injecting into Hydrological Pipeline...")
    grid = Grid.from_raster(tif_filename)

    with rasterio.open(tif_filename) as src:
        dem_data = src.read(1)
        row_idx, col_idx = src.index(target_lon, target_lat)
        target_altitude = dem_data[row_idx, col_idx]
        
        cellsize_x = abs(src.transform[0]) * 111000 * np.cos(np.radians(target_lat))
        cellsize_y = abs(src.transform[4]) * 111000

        try:
            z = dem_data[row_idx-1:row_idx+2, col_idx-1:col_idx+2]
            if z.shape == (3, 3):
                dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0])) / (8 * cellsize_x)
                dz_dy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2])) / (8 * cellsize_y)
                
                slope_rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
                slope_degrees = np.degrees(np.arctan(slope_rise_run))
                slope_percent = slope_rise_run * 100
            else:
                slope_degrees, slope_percent = 0.0, 0.0
        except IndexError:
            slope_degrees, slope_percent = 0.0, 0.0

    print("Conditioning surface topography...")
    dem_raster = grid.read_raster(tif_filename)
    pit_filled = grid.fill_pits(dem_raster)
    flooded = grid.fill_depressions(pit_filled)
    conditioned_dem = grid.resolve_flats(flooded)

    print("Computing flow directions (D8 Routing)...")
    fdir = grid.flowdir(conditioned_dem)

    print("Accumulating grid weights...")
    acc = grid.accumulation(fdir)

    target_acc = acc[row_idx, col_idx]
    log_target_acc = np.log10(target_acc + 1)

    print("Analysis successfully calculated!")

    if log_target_acc < 1.5:
        risk_status = "SAFE"
        risk_desc = "Low accumulation area. Water sheds away naturally. Safe from water logging."
    elif 1.5 <= log_target_acc < 3.0:
        risk_status = "MINOR RISK"
        risk_desc = "Minor collection channel or secondary swale. Watch for short-term pooling during storms."
    else:
        risk_status = "HIGH RISK"
        risk_desc = "Critical accumulation line or terrain sink. Severe risk of pooling, soil anoxia, and flash flow."

    if slope_degrees > 5.0:
        slope_desc = "Steep terrain. High risk of soil erosion, nutrient washing, and rapid runoff velocity."
    elif 2.0 <= slope_degrees <= 5.0:
        slope_desc = "Moderate slope. Good drainage balance; minimal erosion concern under normal conditions."
    else:
        slope_desc = "Flat plain topography. Water moves slowly, maximizing infiltration but increasing pooling vulnerability."

    report_text = f"""
    [LOCATION INFORMATION]            
    Target Coordinates : Lat {target_lat:.5f}, Lon {target_lon:.5f}
    Local Elevation    : {target_altitude:.2f} meters above sea level
    Surface Steepness  : {slope_degrees:.2f}[degrees] ({slope_percent:.1f}% grade)
    Terrain Profile    : {slope_desc}
    [FLOOD RISK ASSESSMENT REPORT]            
    RAW FLOW ACCUM.    : {int(target_acc)} upstream contributing cells
    OVERALL RISK LEVEL : {risk_status}
    Risk Assessment    : {risk_desc}
    """
    print(report_text)
    return report_text


def _calculate_flood_risk(tif_path) -> str:
    center_coords = _extract_tif_midpoint(tif_path)
    target_lat = center_coords["center_coords"]["lat"]
    target_lon = center_coords["center_coords"]["lon"]
    bbox = _bbox_from_point(target_lat, target_lon)
    dem_folder = r"C:\Users\malad\OneDrive\Device\Makeathon\github_repo\makeathon-2026-panick-attack-real-estate-beyond-rgb\data\dem"
    local_tif = os.path.join(dem_folder, "output_hh.tif")
    temp_clipped_tif = "temp_greece_dem.tif"
    
    print(f"Reading and clipping local DEM: {local_tif}...")
    
    # Open the local raster with rioxarray to perform the spatial clip
    rds = rioxarray.open_rasterio(local_tif)
    
    # Create a bounding box geometry for the clip
    geom = shapely.geometry.box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    
    # Clip the raster to your bounding box area
    # (crs="EPSG:4326" assumes your bbox coordinates match the TIF's coordinate system)
    clipped_rds = rds.rio.clip([geom], crs="EPSG:4326")
    
    # Save the clipped region to match your downstream workflow filename
    clipped_rds.rio.to_raster(temp_clipped_tif)

    print("DEM Clipping Complete. Injecting into Hydrological Pipeline...")
    grid = Grid.from_raster(temp_clipped_tif)

    with rasterio.open(temp_clipped_tif) as src:
        dem_data = src.read(1)
        # Note: rasterio expects (lon, lat) for indexing
        row_idx, col_idx = src.index(target_lon, target_lat)
        target_altitude = dem_data[row_idx, col_idx]
        
        cellsize_x = abs(src.transform[0]) * 111000 * np.cos(np.radians(target_lat))
        cellsize_y = abs(src.transform[4]) * 111000

        try:
            z = dem_data[row_idx-1:row_idx+2, col_idx-1:col_idx+2]
            if z.shape == (3, 3):
                dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0])) / (8 * cellsize_x)
                dz_dy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2])) / (8 * cellsize_y)
                
                slope_rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
                slope_degrees = np.degrees(np.arctan(slope_rise_run))
                slope_percent = slope_rise_run * 100
            else:
                slope_degrees, slope_percent = 0.0, 0.0
        except IndexError:
            slope_degrees, slope_percent = 0.0, 0.0

    print("Conditioning surface topography...")
    dem_raster = grid.read_raster(temp_clipped_tif)
    pit_filled = grid.fill_pits(dem_raster)
    flooded = grid.fill_depressions(pit_filled)
    conditioned_dem = grid.resolve_flats(flooded)

    print("Computing flow directions (D8 Routing)...")
    fdir = grid.flowdir(conditioned_dem)

    print("Accumulating grid weights...")
    acc = grid.accumulation(fdir)

    target_acc = acc[row_idx, col_idx]
    log_target_acc = np.log10(target_acc + 1)

    print("Analysis successfully calculated!")

    if log_target_acc < 1.5:
        risk_status = "SAFE"
        risk_desc = "Low accumulation area. Water sheds away naturally. Safe from water logging."
    elif 1.5 <= log_target_acc < 3.0:
        risk_status = "MINOR RISK"
        risk_desc = "Minor collection channel or secondary swale. Watch for short-term pooling during storms."
    else:
        risk_status = "HIGH RISK"
        risk_desc = "Critical accumulation line or terrain sink. Severe risk of pooling, soil anoxia, and flash flow."

    if slope_degrees > 5.0:
        slope_desc = "Steep terrain. High risk of soil erosion, nutrient washing, and rapid runoff velocity."
    elif 2.0 <= slope_degrees <= 5.0:
        slope_desc = "Moderate slope. Good drainage balance; minimal erosion concern under normal conditions."
    else:
        slope_desc = "Flat plain topography. Water moves slowly, maximizing infiltration but increasing pooling vulnerability."

    report_text = f"""
    [LOCATION INFORMATION]            
    Target Coordinates : Lat {target_lat:.5f}, Lon {target_lon:.5f}
    Local Elevation    : {target_altitude:.2f} meters above sea level
    Surface Steepness  : {slope_degrees:.2f}[degrees] ({slope_percent:.1f}% grade)
    Terrain Profile    : {slope_desc}
    [FLOOD RISK ASSESSMENT REPORT]            
    RAW FLOW ACCUM.    : {int(target_acc)} upstream contributing cells
    OVERALL RISK LEVEL : {risk_status}
    Risk Assessment    : {risk_desc}
    """
    print(report_text)
    
    # Optional clean up of temporary clipped file
    if os.path.exists(temp_clipped_tif):
        try:
            os.remove(temp_clipped_tif)
        except Exception:
            pass # Keep going if file handles are still locked by pysheds/rasterio
            
    return report_text

def _calculate_road_proximity(target_lat: float, target_lon: float) -> str:
    print("Querying OpenStreetMap directly from coordinate center point...")
    try:
        roads_graph = ox.graph.graph_from_point(
            center_point=(target_lat, target_lon),
            dist=5000,
            network_type="drive",
            simplify=True,
            retain_all=True
        )
        
        roads_graph_projected = ox.projection.project_graph(roads_graph)
        _, edges = ox.graph_to_gdfs(roads_graph_projected)
        target_point_wgs = Point(target_lon, target_lat)
        
        target_point_projected = ox.projection.project_geometry(
            target_point_wgs, 
            to_crs=roads_graph_projected.graph["crs"]
        )[0]
        
        road_distance_meters = edges.distance(target_point_projected).min()
        # road_point = edges.geometry[edges.distance(target_point_projected).idxmin()]
        road_report_text = f"""
    [Accessibility / ROAD PROXIMITY REPORT]       
    Road Proximity   : {road_distance_meters:.2f} meters to nearest drivable road
        """     
        print(road_report_text)
        return road_report_text 
        
    except Exception as e:
        print(f"Direct coordinate point proximity analysis failed: {e}")
        return f"Road Proximity Analysis Failed: {str(e)}"


def _process_enmap_soil_data(input_file: str) -> tuple[str, pd.DataFrame]:
    """Processes hyperspectral TIF data, auto-correcting scale factors, clamping 
    outlier pixels, and calculating all macronutrients uniformly in mg/kg.
    """
    df = _tif_to_dataframe(input_file)
    
    band_cols = [col for col in df.columns if col.startswith('Band_')]
    print(f"Detected {len(band_cols)} spectral bands. Interpolating atmospheric gaps...")
    
    df_cleaned = df.copy()
    
    # FIX 1: Convert black mosaic edge zeros to NaN so they don't break spatial averages
    df_cleaned[band_cols] = df_cleaned[band_cols].replace(0, np.nan)
    
    # Interpolate across the bands
    df_cleaned[band_cols] = df_cleaned[band_cols].interpolate(axis=1, limit_direction='both')
    
    # Fill remaining edge gaps with a standard baseline fraction
    df_cleaned[band_cols] = df_cleaned[band_cols].fillna(0.2)
    df_cleaned[band_cols] = df_cleaned[band_cols].clip(lower=0.0, upper=1.0)
    
    np.random.seed(42)
    num_samples = len(df_cleaned)
    
    print("Computing soil chemical and organic properties...")
    # Your original spectral band means extraction
    vis_mean = df_cleaned[[f'Band_{i}' for i in range(1, 41) if f'Band_{i}' in df_cleaned.columns]].mean(axis=1)
    
    # Organic matter base profiles
    predicted_som = 6.5 - (vis_mean * 3.5) + np.random.normal(0, 0.1, size=num_samples)
    predicted_soc = predicted_som / 1.724
    
    # Original Nitrogen logic (percentage baseline)
    predicted_n_pct = predicted_som * 0.075 + np.random.normal(0, 0.015, size=num_samples)
    # CONVERSION: Convert the raw percentage profile into mg/kg (1% = 10,000 mg/kg)
    predicted_n_mg_kg = predicted_n_pct * 10000.0
    
    swir_ratio = df_cleaned['Band_200'] / (df_cleaned['Band_100'] + 1e-5)
    predicted_ph = 5.2 + (swir_ratio * 1.6) + np.random.normal(0, 0.15, size=num_samples)
    
    nir_mean = df_cleaned[[f'Band_{i}' for i in range(45, 85) if f'Band_{i}' in df_cleaned.columns]].mean(axis=1)
    predicted_p = 12.0 + (nir_mean * 15) - (vis_mean * 8) + np.random.normal(0, 1.5, size=num_samples)
    
    swir2_mean = df_cleaned[[f'Band_{i}' for i in range(150, 220) if f'Band_{i}' in df_cleaned.columns]].mean(axis=1)
    predicted_k = 90.0 + (swir2_mean * 180) - (vis_mean * 40) + np.random.normal(0, 10, size=num_samples)
    
    swir1_mean = df_cleaned[[f'Band_{i}' for i in range(100, 150) if f'Band_{i}' in df_cleaned.columns]].mean(axis=1)
    predicted_mg = 50.0 + (swir1_mean * 120) - (vis_mean * 25) + np.random.normal(0, 5, size=num_samples)
   
    ndvi = (df_cleaned['Band_80'] - df_cleaned['Band_40']) / (df_cleaned['Band_80'] + df_cleaned['Band_40'] + 1e-5)
    swi = df_cleaned['Band_200'] / (df_cleaned['Band_80'] + 1e-5)

    output_df = pd.DataFrame({
        'Pixel_ID': df_cleaned['Pixel_ID'],
        'pH_Assessment': np.round(predicted_ph, 2),
        'Nitrogen_N_mg_kg': np.round(predicted_n_mg_kg, 1),  # Stored cleanly as mg/kg now
        'Phosphorus_P_mg_kg': np.round(predicted_p, 1),
        'Potassium_K_mg_kg': np.round(predicted_k, 1),
        'Magnesium_Mg_mg_kg': np.round(predicted_mg, 1),
        'SOM_pct': np.round(predicted_som, 2),
        'SOC_pct': np.round(predicted_soc, 2),
        'NDVI': np.round(ndvi, 3),
        'SWI': np.round(swi, 3)
    })
    
    summary_stats = output_df.describe().transpose()
    ph_mean = summary_stats.loc['pH_Assessment', 'mean']
    som_mean = summary_stats.loc['SOM_pct', 'mean']
    soc_mean = summary_stats.loc['SOC_pct', 'mean']

    # PH Interpretations
    if ph_mean < 5.0:
        ph_status = "Very Acidic"
    elif 5.0 <= ph_mean < 6.0:
        ph_status = "Slightly Acidic"
    elif 6.0 <= ph_mean <= 7.2:
        ph_status = "Optimal / Near-Neutral"
    else:
        ph_status = "Alkaline"
    
    # SOM Interpretations
    if som_mean > 3.0:
        som_status = "Excellent Structural Resilience"
    else:
        som_status = "Low Organic Buffer"
    
    # Safe mean metrics extractions
    n_mean = summary_stats.loc['Nitrogen_N_mg_kg', 'mean']
    p_mean = summary_stats.loc['Phosphorus_P_mg_kg', 'mean']
    k_mean = summary_stats.loc['Potassium_K_mg_kg', 'mean']
    mg_mean = summary_stats.loc['Magnesium_Mg_mg_kg', 'mean']

    def classify_level(value, low, moderate, high):
        if value < low:
            return "DEFICIENT"
        elif value < moderate:
            return "LOW"
        elif value < high:
            return "ADEQUATE"
        else:
            return "HIGH"


    # --- Mediterranean-oriented thresholds ---

    # Total Nitrogen (mg/kg)
    n_status = classify_level(
        n_mean,
        low=1000.0,
        moderate=2000.0,
        high=3500.0
    )

    # Available Phosphorus (mg/kg)
    # Appropriate for Mediterranean calcareous soils using Olsen-P interpretation
    p_status = classify_level(
        p_mean,
        low=10.0,
        moderate=20.0,
        high=35.0
    )

    # Exchangeable Potassium (mg/kg)
    k_status = classify_level(
        k_mean,
        low=80.0,
        moderate=150.0,
        high=250.0
    )

    # Exchangeable Magnesium (mg/kg)
    mg_status = classify_level(
        mg_mean,
        low=50.0,
        moderate=100.0,
        high=250.0
    )

    soil_report_text = f"""
    [CHEMICAL CHARACTERISTICS]
    Soil pH Profile     : {ph_mean:.2f} ({ph_status})  [Range: {summary_stats.loc['pH_Assessment', 'min']:.2f} - {summary_stats.loc['pH_Assessment', 'max']:.2f}]
    [ORGANIC MATTER & CARBON STRUCTURE]
    Soil Organic Matter : {som_mean:.2f}% ({som_status})
    Soil Organic Carbon : {soc_mean:.2f}%
    [MACRONUTRIENT RESERVES ANALYSIS]
    Total Nitrogen (N)  : {n_mean:.1f} mg/kg  ({n_status})
    Phosphorus (P)      : {p_mean:.1f} mg/kg  ({p_status})
    Potassium (K)       : {k_mean:.1f} mg/kg  ({k_status})
    Magnesium (Mg)      : {mg_mean:.1f} mg/kg  ({mg_status})
    """
    print(soil_report_text)
    return soil_report_text, output_df

@mcp.tool()
def plot_soil_analysis_report(tif_path):
    """Generates a professional visual dashboard for the calculated soil metrics."""
    _, output_df = _process_enmap_soil_data(tif_path)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 16
    })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("HYPERSPECTRAL SOIL DIAGNOSTIC", fontweight="bold", y=0.96)
    ax = axes.flatten()

    colors = {
        "pH": "#4f81bd", "N": "#c0504d", "P": "#9bbb59",
        "K": "#8064a2", "Mg": "#4bacc6", "SOM": "#f79646"
    }

    # Plots
    sns.histplot(data=output_df, x="pH_Assessment", kde=True, ax=ax[0], color=colors["pH"], edgecolor="black", alpha=0.7)
    ax[0].axvline(output_df["pH_Assessment"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['pH_Assessment'].mean():.2f}")
    ax[0].set_title("Soil pH Profile", fontweight="bold")
    ax[0].legend()

    sns.histplot(data=output_df, x="Nitrogen_N_mg_kg", kde=True, ax=ax[1], color=colors["N"], edgecolor="black", alpha=0.7)
    ax[1].axvline(output_df["Nitrogen_N_mg_kg"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['Nitrogen_N_mg_kg'].mean():.1f} mg/kg")
    ax[1].set_title("Total Nitrogen (N %)", fontweight="bold")
    ax[1].legend()

    sns.histplot(data=output_df, x="Phosphorus_P_mg_kg", kde=True, ax=ax[2], color=colors["P"], edgecolor="black", alpha=0.7)
    ax[2].axvline(output_df["Phosphorus_P_mg_kg"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['Phosphorus_P_mg_kg'].mean():.1f} mg/kg")
    ax[2].set_title("Available Phosphorus (P)", fontweight="bold")
    ax[2].legend()


    sns.histplot(data=output_df, x="Potassium_K_mg_kg", kde=True, ax=ax[3], color=colors["K"], edgecolor="black", alpha=0.7)
    ax[3].axvline(output_df["Potassium_K_mg_kg"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['Potassium_K_mg_kg'].mean():.1f} mg/kg")
    ax[3].set_title("Exchangeable Potassium (K)", fontweight="bold")
    ax[3].legend()

    sns.histplot(data=output_df, x="Magnesium_Mg_mg_kg", kde=True, ax=ax[4], color=colors["Mg"], edgecolor="black", alpha=0.7)
    ax[4].axvline(output_df["Magnesium_Mg_mg_kg"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['Magnesium_Mg_mg_kg'].mean():.1f} mg/kg")
    ax[4].set_title("Magnesium (Mg)", fontweight="bold")
    ax[4].legend()

    sns.histplot(data=output_df, x="SOM_pct", kde=True, ax=ax[5], color=colors["SOM"], edgecolor="black", alpha=0.7)
    ax[5].axvline(output_df["SOM_pct"].mean(), color="darkred", linestyle="--", label=f"Mean: {output_df['SOM_pct'].mean():.1f}%")
    ax[5].set_title("Soil Organic Matter (SOM %)", fontweight="bold")
    ax[5].legend()

    scatter = ax[6].scatter(output_df["NDVI"], output_df["SWI"], c=output_df["pH_Assessment"], cmap="viridis", alpha=0.6, s=25)
    ax[6].set_title("NDVI vs SWI (by pH)", fontweight="bold")
    ax[6].set_xlabel("NDVI")
    ax[6].set_ylabel("SWI")
    cbar = fig.colorbar(scatter, ax=ax[6], orientation="vertical", shrink=0.7)
    cbar.set_label("pH", size=9)

    ax[7].axis("off")
    summary_text = (
        f"=== AGROMANAGEMENT METRICS ===\n\n"
        f"Total Samples Bound : {len(output_df)} pixels\n"
        f"Mean Soil pH        : {output_df['pH_Assessment'].mean():.2f}\n"
        f"Mean SOM Buffer     : {output_df['SOM_pct'].mean():.2f}%\n"
        f"Mean Nitrogen (N)   : {output_df['Nitrogen_N_mg_kg'].mean():.3f} mg/kg\n"
        f"Mean Phosphorus (P) : {output_df['Phosphorus_P_mg_kg'].mean():.1f} mg/kg\n"
        f"Mean Potassium (K)  : {output_df['Potassium_K_mg_kg'].mean():.1f} mg/kg\n"
        f"Mean NDVI Baseline  : {output_df['NDVI'].mean():.3f}\n"
        f"Mean Moisture (SWI) : {output_df['SWI'].mean():.3f}\n\n"
    )
    ax[7].text(0.05, 0.95, summary_text, transform=ax[7].transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#ccc"))
    
    
    ax[8].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    savefig_path = f"SOIL_DIAGNOSTIC_REPORT{os.path.basename(tif_path)}.png"
    plt.savefig(savefig_path, dpi=300, bbox_inches="tight")
    print(f"Report saved successfully to: {savefig_path}")
    plt.close()

@mcp.tool()
def CREATE_GEO_AND_RISK_REPORT(TIF_DATA_PATH: str) -> str:
    """Generate a comprehensive geo-spatial and risk report based on the provided GeoTIFF file."""
    midpoint_info = _extract_tif_midpoint(TIF_DATA_PATH)
    if not midpoint_info.get("success"):
        return f"GeoTIFF midpoint extraction failed: {midpoint_info.get('error', 'unknown error')}"

    target_lat = midpoint_info["center_coords"]["lat"]
    target_lon = midpoint_info["center_coords"]["lon"]

    print(f"Extracted midpoint coordinates from GeoTIFF: Latitude {target_lat}, Longitude {target_lon}")
    report = ''
    report += _calculate_flood_risk(target_lat, target_lon)
    report += _calculate_road_proximity(target_lat, target_lon)
    report_soil, _ = _process_enmap_soil_data(TIF_DATA_PATH)    
    report += report_soil
    return report.upper()

if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="streamable-http")