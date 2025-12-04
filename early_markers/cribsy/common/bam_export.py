"""Excel export utilities for BAM results.

This module provides functions to export BAM results to professionally
formatted Excel workbooks suitable for grants, regulatory submissions,
and stakeholder communication.

Example:
    >>> from early_markers.cribsy.common.bam_unified import BAMEstimator
    >>> from early_markers.cribsy.common.bam_export import export_bam_results_to_excel
    >>> 
    >>> pilot = np.array([1]*15 + [0]*5)
    >>> estimator = BAMEstimator(seed=42)
    >>> result = estimator.estimate_single(pilot, target_width=0.15)
    >>> 
    >>> export_bam_results_to_excel(
    ...     [result],
    ...     "bam_analysis.xlsx",
    ...     title="Sample Size Analysis"
    ... )

Author: AI Assistant
Date: 2025-01-05
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from xlsxwriter import Workbook
except ImportError:
    Workbook = None


def _check_dependencies():
    """Check if required dependencies are available."""
    if pd is None:
        raise ImportError(
            "Excel export requires pandas. Install with: pip install pandas"
        )
    if Workbook is None:
        raise ImportError(
            "Excel export requires xlsxwriter. Install with: pip install xlsxwriter"
        )


def _create_workbook_formats(wb: Any) -> Dict[str, Any]:
    """Create standard formatting styles for workbook.
    
    Args:
        wb: xlsxwriter Workbook object.
    
    Returns:
        Dictionary of format objects.
    """
    # Color palette
    DK_BLUE = "#4F81BD"
    MD_BLUE = "#95B3D7"
    LT_BLUE = "#DCE6F1"
    
    # Header formats
    bold_fmt = wb.add_format({"bold": True})
    
    head_fmt = wb.add_format({"bold": True, "font_size": 13})
    head_fmt.set_align("left")
    
    hdr_fmt = wb.add_format({"bold": True, "font_color": "white"})
    hdr_fmt.set_align("center")
    hdr_fmt.set_bg_color(DK_BLUE)
    
    # Data formats
    code_fmt = wb.add_format({"bold": True})
    code_fmt.set_bg_color(MD_BLUE)
    
    desc_fmt = wb.add_format()
    desc_fmt.set_bg_color(LT_BLUE)
    
    # Number formats
    pct_fmt = wb.add_format({"num_format": "0.00%"})
    
    blue_pct_fmt = wb.add_format({"num_format": "0.00%"})
    blue_pct_fmt.set_bg_color(LT_BLUE)
    
    dec2_fmt = wb.add_format({"num_format": "0.00"})
    
    dec3_fmt = wb.add_format({"num_format": "0.000"})
    
    return {
        "bold": bold_fmt,
        "heading": head_fmt,
        "header": hdr_fmt,
        "code": code_fmt,
        "desc": desc_fmt,
        "pct": pct_fmt,
        "blue_pct": blue_pct_fmt,
        "dec2": dec2_fmt,
        "dec3": dec3_fmt,
    }


def export_bam_results_to_excel(
    results: List[Any],  # List[BAMResult]
    output_path: Path,
    include_plots: bool = False,
    title: str = "BAM Sample Size Analysis"
):
    """Export BAM results to formatted Excel workbook.
    
    Creates multi-sheet workbook with:
    - Summary table of all results
    - Detailed sheets for each result
    - Metadata sheet with parameters
    
    Args:
        results: List of BAMResult objects from BAM estimation.
        output_path: Path to save Excel file (.xlsx).
        include_plots: Whether to embed plots as images (requires Pillow).
        title: Title for workbook and summary sheet.
    
    Example:
        >>> from early_markers.cribsy.common.bam_unified import BAMEstimator
        >>> pilot = np.array([1]*15 + [0]*5)
        >>> estimator = BAMEstimator(seed=42)
        >>> 
        >>> results = []
        >>> for width in [0.10, 0.15, 0.20]:
        ...     result = estimator.estimate_single(pilot, target_width=width)
        ...     results.append(result)
        >>> 
        >>> export_bam_results_to_excel(results, "sample_sizes.xlsx")
    """
    _check_dependencies()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wb = Workbook(str(output_path))
    formats = _create_workbook_formats(wb)
    
    # Sheet 1: Summary
    _write_summary_sheet(wb, results, formats, title)
    
    # Sheet 2+: Individual results
    for i, result in enumerate(results, 1):
        sheet_name = f"Result_{i}_{result.metric_type}"
        ws = wb.add_worksheet(sheet_name[:31])  # Excel max 31 chars
        _write_result_sheet(ws, result, formats, include_plots)
    
    # Sheet N: Metadata
    _write_metadata_sheet(wb, results, formats)
    
    wb.close()
    
    print(f"Exported {len(results)} BAM results to: {output_path}")


def _write_summary_sheet(wb, results, formats, title):
    """Write summary sheet with all results in table format."""
    ws = wb.add_worksheet("Summary")
    
    # Title
    ws.merge_range(0, 0, 0, 6, title, formats["heading"])
    
    # Headers
    row = 2
    headers = [
        "Result #",
        "Metric Type",
        "Optimal N",
        "Target Width",
        "Target Assurance",
        "Achieved Assurance",
        "Pilot Estimate",
        "ICC",
        "Cluster Size",
        "Design Effect",
        "Computation (s)",
    ]
    
    for col, header in enumerate(headers):
        ws.write(row, col, header, formats["header"])
    
    # Data rows
    for i, result in enumerate(results, 1):
        row += 1
        ws.write(row, 0, i, formats["bold"])
        ws.write(row, 1, result.metric_type)
        ws.write(row, 2, result.optimal_n, formats["bold"])
        ws.write(row, 3, result.target_width, formats["dec2"])
        ws.write(row, 4, result.target_assurance, formats["pct"])
        ws.write(row, 5, result.achieved_assurance, formats["pct"])
        ws.write(row, 6, result.pilot_estimate, formats["dec3"])
        ws.write(row, 7, result.icc if result.icc else "N/A", formats["dec3"] if result.icc else None)
        ws.write(row, 8, result.mean_cluster_size if result.mean_cluster_size else "N/A")
        ws.write(row, 9, result.design_effect if result.design_effect else "N/A", formats["dec3"] if result.design_effect else None)
        ws.write(row, 10, result.computation_time, formats["dec2"])
    
    # Conditional formatting for optimal N
    ws.conditional_format(3, 2, row, 2, {
        "type": "3_color_scale",
        "min_color": "#63be7b",
        "mid_color": "#ffeb83",
        "max_color": "#f8696b",
    })
    
    # Auto-fit columns
    ws.set_column(0, 0, 10)
    ws.set_column(1, 1, 18)
    ws.set_column(2, 2, 12)
    ws.set_column(3, 10, 15)
    
    # Freeze top rows
    ws.freeze_panes(3, 0)


def _write_result_sheet(ws, result, formats, include_plots):
    """Write detailed sheet for individual result."""
    row = 0
    
    # Result header
    ws.merge_range(row, 0, row, 3, f"BAM Result Details - {result.metric_type.replace('_', ' ').title()}", formats["heading"])
    row += 2
    
    # Key results
    ws.write(row, 0, "Optimal N:", formats["code"])
    ws.write(row, 1, result.optimal_n, formats["bold"])
    row += 1
    
    ws.write(row, 0, "Target Width:", formats["desc"])
    ws.write(row, 1, result.target_width, formats["dec2"])
    row += 1
    
    ws.write(row, 0, "Target Assurance:", formats["desc"])
    ws.write(row, 1, result.target_assurance, formats["pct"])
    row += 1
    
    ws.write(row, 0, "Achieved Assurance:", formats["code"])
    ws.write(row, 1, result.achieved_assurance, formats["pct"])
    row += 1
    
    ws.write(row, 0, "Pilot Estimate:", formats["desc"])
    ws.write(row, 1, result.pilot_estimate, formats["dec3"])
    row += 1
    
    ws.write(row, 0, "CI Level:", formats["desc"])
    ws.write(row, 1, result.ci_level, formats["pct"])
    row += 2
    
    # Cluster info if present
    if result.icc is not None:
        ws.write(row, 0, "ICC:", formats["code"])
        ws.write(row, 1, result.icc, formats["dec3"])
        row += 1
        
        ws.write(row, 0, "Mean Cluster Size:", formats["desc"])
        ws.write(row, 1, result.mean_cluster_size, formats["dec2"])
        row += 1
        
        ws.write(row, 0, "Design Effect:", formats["desc"])
        ws.write(row, 1, result.design_effect, formats["dec3"])
        row += 1
        
        ws.write(row, 0, "Total N (clusters × size):", formats["code"])
        ws.write(row, 1, int(result.optimal_n * result.mean_cluster_size), formats["bold"])
        row += 2
    
    # Computation details
    ws.write(row, 0, "Computation Time (s):", formats["desc"])
    ws.write(row, 1, result.computation_time, formats["dec2"])
    row += 1
    
    ws.write(row, 0, "Search Iterations:", formats["desc"])
    ws.write(row, 1, result.search_iterations)
    row += 1
    
    ws.write(row, 0, "Simulations per N:", formats["desc"])
    ws.write(row, 1, result.simulations_per_n)
    row += 2
    
    # Assurance curve data
    ws.merge_range(row, 0, row, 3, "Assurance Curve Data", formats["heading"])
    row += 1
    
    # Headers
    ws.write(row, 0, "Sample Size", formats["header"])
    ws.write(row, 1, "Assurance", formats["header"])
    row += 1
    
    # Data
    for n, assurance in zip(result.sample_sizes_tested, result.assurances_at_tested):
        ws.write(row, 0, n)
        ws.write(row, 1, assurance, formats["pct"])
        row += 1
    
    # Auto-fit
    ws.set_column(0, 0, 25)
    ws.set_column(1, 1, 15)


def _write_metadata_sheet(wb, results, formats):
    """Write metadata sheet with analysis parameters."""
    ws = wb.add_worksheet("Metadata")
    
    row = 0
    ws.merge_range(row, 0, row, 1, "Analysis Metadata", formats["heading"])
    row += 2
    
    ws.write(row, 0, "Total Results:", formats["code"])
    ws.write(row, 1, len(results), formats["bold"])
    row += 2
    
    ws.write(row, 0, "Metric Types:", formats["desc"])
    row += 1
    metric_counts = {}
    for result in results:
        metric_counts[result.metric_type] = metric_counts.get(result.metric_type, 0) + 1
    for metric, count in metric_counts.items():
        ws.write(row, 0, f"  {metric}", formats["desc"])
        ws.write(row, 1, count)
        row += 1
    
    row += 1
    ws.write(row, 0, "Software:", formats["desc"])
    ws.write(row, 1, "BAM Unified API (early-markers)")
    row += 1
    
    ws.write(row, 0, "Method:", formats["desc"])
    ws.write(row, 1, "Bayesian Assurance Method")
    row += 1
    
    ws.set_column(0, 0, 25)
    ws.set_column(1, 1, 30)


def export_grid_to_excel(
    grid_df,
    output_path: Path,
    pivot_by: str = "target_width",
    title: str = "BAM Grid Search Results"
):
    """Export grid search results to formatted Excel with pivot tables.
    
    Args:
        grid_df: DataFrame from bam_grid_search() or bam_grid_search_joint().
        output_path: Path to save Excel file (.xlsx).
        pivot_by: Column to pivot by for summary tables.
        title: Title for workbook.
    
    Example:
        >>> from early_markers.cribsy.common.bam_unified import bam_grid_search
        >>> pilot = np.array([1]*15 + [0]*5)
        >>> results = bam_grid_search(
        ...     pilot,
        ...     target_widths=[0.05, 0.10, 0.15, 0.20],
        ...     target_assurances=[0.75, 0.80, 0.85],
        ...     n_jobs=-1
        ... )
        >>> export_grid_to_excel(results, "grid_results.xlsx")
    """
    _check_dependencies()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wb = Workbook(str(output_path))
    formats = _create_workbook_formats(wb)
    
    # Sheet 1: Raw data
    ws_raw = wb.add_worksheet("Raw Data")
    ws_raw.write(0, 0, title, formats["heading"])
    
    # Write dataframe
    for col_num, col_name in enumerate(grid_df.columns):
        ws_raw.write(2, col_num, col_name, formats["header"])
    
    for row_num, row_data in enumerate(grid_df.itertuples(index=False), start=3):
        for col_num, value in enumerate(row_data):
            if isinstance(value, float):
                ws_raw.write(row_num, col_num, value, formats["dec3"])
            else:
                ws_raw.write(row_num, col_num, value)
    
    ws_raw.freeze_panes(3, 0)
    
    # Sheet 2: Pivot table (if possible)
    if 'target_assurance' in grid_df.columns and 'optimal_n' in grid_df.columns:
        ws_pivot = wb.add_worksheet("Pivot Table")
        ws_pivot.write(0, 0, f"{title} - Pivot by {pivot_by}", formats["heading"])
        
        try:
            pivot = grid_df.pivot_table(
                index='target_assurance',
                columns=pivot_by,
                values='optimal_n',
                aggfunc='first'
            )
            
            # Write pivot table
            row = 2
            # Column headers
            ws_pivot.write(row, 0, "Target Assurance", formats["header"])
            for col_num, col_val in enumerate(pivot.columns, start=1):
                ws_pivot.write(row, col_num, f"{col_val}", formats["header"])
            
            # Data
            for row_num, (idx, row_data) in enumerate(pivot.iterrows(), start=3):
                ws_pivot.write(row_num, 0, idx, formats["pct"])
                for col_num, val in enumerate(row_data, start=1):
                    ws_pivot.write(row_num, col_num, val if pd.notna(val) else "N/A", formats["bold"] if pd.notna(val) else None)
            
            # Conditional formatting
            max_row = row + len(pivot)
            max_col = len(pivot.columns)
            ws_pivot.conditional_format(3, 1, max_row, max_col, {
                "type": "3_color_scale",
                "min_color": "#63be7b",
                "mid_color": "#ffeb83",
                "max_color": "#f8696b",
            })
            
            ws_pivot.freeze_panes(3, 1)
        except Exception as e:
            warnings.warn(f"Could not create pivot table: {e}")
    
    wb.close()
    
    print(f"Exported grid search results to: {output_path}")
