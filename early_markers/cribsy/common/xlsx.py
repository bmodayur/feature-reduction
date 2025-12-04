"""Excel formatting utilities for early-markers reports.

This module provides standardized formatting functions for creating
professional-looking Excel reports with consistent styling across
the early-markers project.

Functions:
    set_workbook_formats: Create a dictionary of reusable Excel formats

Color Scheme:
    - Blue: Primary theme (headings, headers)
    - Green: Highlighting optimal/best values
    - Tan: Alternative highlighting
    - Gray: Lists and secondary content

Example:
    >>> from xlsxwriter import Workbook
    >>> from early_markers.cribsy.common.xlsx import set_workbook_formats
    >>>
    >>> wb = Workbook('report.xlsx')
    >>> formats = set_workbook_formats(wb)
    >>> ws = wb.add_worksheet('Summary')
    >>> ws.write('A1', 'Title', formats['heading1'])
    >>> wb.close()
"""
from xlsxwriter import Workbook


def set_workbook_formats(wb: Workbook) -> dict:
    """Create a dictionary of reusable Excel cell formats.

    Generates a comprehensive set of pre-configured XlsxWriter format objects
    for consistent styling across Excel reports. Includes formats for headings,
    data cells, percentages, and conditional formatting backgrounds.

    Args:
        wb (Workbook): XlsxWriter Workbook object to attach formats to.

    Returns:
        dict[str, Format]: Dictionary mapping format names to Format objects.
            Available format keys:
            - 'bold': Bold text
            - 'heading1': Large heading (22pt, bold, left-aligned)
            - 'heading2': Medium heading (18pt, bold, left-aligned)
            - 'heading3': Small heading (14pt, bold, left-aligned)
            - 'header': Table header (bold, centered, dark blue background)
            - 'code': Code/identifier (bold, medium blue background)
            - 'desc': Description (light blue background)
            - 'pct': Percentage format (0.00%)
            - 'blue_pct': Percentage with light blue background
            - 'green_bg': Green background (for optimal values)
            - 'blue_bg': Blue background
            - 'tan_bg': Tan background
            - 'gray_bg': Gray background
            - 'long_list': Wrapped text list (top-aligned, gray)
            - 'long_list_heading': List heading (18pt, bold, gray)

    Color Palette:
        Blues: #4F81BD (dark), #95B3D7 (medium), #DCE6F1 (light)
        Tans: #948a54 (dark), #c4bd97 (medium), #dddac4 (light)
        Grays: #a6a6a6 (dark), #bfbfbf (medium), #d9d9d9 (light)
        Green: #63be7b (highlighting)

    Example:
        >>> from xlsxwriter import Workbook
        >>> from early_markers.cribsy.common.xlsx import set_workbook_formats
        >>>
        >>> # Create workbook and get formats
        >>> wb = Workbook('analysis_report.xlsx')
        >>> fmts = set_workbook_formats(wb)
        >>> ws = wb.add_worksheet('Results')
        >>>
        >>> # Use formats
        >>> ws.write('A1', 'Analysis Summary', fmts['heading1'])
        >>> ws.write('A2', 'Metric', fmts['header'])
        >>> ws.write('B2', 'Value', fmts['header'])
        >>> ws.write('A3', 'Sensitivity', fmts['desc'])
        >>> ws.write('B3', 0.8547, fmts['pct'])
        >>>
        >>> # Conditional formatting for optimal values
        >>> ws.conditional_format('B3:B10', {
        ...     'type': 'formula',
        ...     'criteria': '=B3>=0.85',
        ...     'format': fmts['green_bg']
        ... })
        >>>
        >>> wb.close()

    Note:
        Formats must be created before writing any data to the workbook.
        XlsxWriter requires formats to be added to the workbook object
        they will be used with.

    See Also:
        BayesianData.write_excel_report: Uses these formats for ROC reports
    """
    DK_BLUE = "#4F81BD"
    MD_BLUE = "#95B3D7"
    LT_BLUE = "#DCE6F1"

    DK_TAN =  "#948a54"
    MD_TAN =  "#c4bd97"
    LT_TAN =  "#dddac4"

    DK_GRAY = "#a6a6a6"
    MD_GRAY = "#bfbfbf"
    LT_GRAY = "#d9d9d9"

    bold_fmt = wb.add_format({"bold": True})

    head1_fmt = wb.add_format({"bold": True, "font_size": 22})
    head1_fmt.set_align("left")

    head2_fmt = wb.add_format({"bold": True, "font_size": 18})
    head2_fmt.set_align("left")

    head3_fmt = wb.add_format({"bold": True, "font_size": 14})
    head3_fmt.set_align("left")

    hdr_fmt = wb.add_format({"bold": True})
    hdr_fmt.set_align("center")
    hdr_fmt.set_bg_color(DK_BLUE)

    code_fmt = wb.add_format({"bold": True})
    code_fmt.set_bg_color(MD_BLUE)

    desc_fmt = wb.add_format()
    desc_fmt.set_bg_color(LT_BLUE)

    pct_fmt = wb.add_format({"num_format": "0.00%"})
    blue_pct_fmt = wb.add_format({"num_format": "0.00%"})
    blue_pct_fmt.set_bg_color(LT_BLUE)

    green_bg_format = wb.add_format()
    green_bg_format.set_bg_color("#63be7b")
    blue_bg_format = wb.add_format()
    blue_bg_format.set_bg_color("#668dc8")
    tan_bg_format = wb.add_format()
    tan_bg_format.set_bg_color(LT_TAN)
    gray_bg_format = wb.add_format()
    gray_bg_format.set_bg_color(LT_GRAY)

    long_list_fmt = wb.add_format()
    long_list_fmt.set_valign("top")
    long_list_fmt.set_text_wrap(True)
    long_list_fmt.set_bg_color(LT_GRAY)

    long_list_head_fmt = wb.add_format({"bold": True, "font_size": 18})
    long_list_head_fmt.set_bg_color(MD_GRAY)

    return {
        "bold": bold_fmt,
        "heading1": head1_fmt,
        "heading2": head2_fmt,
        "heading3": head3_fmt,
        "header": hdr_fmt,
        "code": code_fmt,
        "desc": desc_fmt,
        "pct": pct_fmt,
        "blue_pct": blue_pct_fmt,
        "green_bg": green_bg_format,
        "blue_bg": blue_bg_format,
        "tan_bg": tan_bg_format,
        "gray_bg": gray_bg_format,
        "long_list": long_list_fmt,
        "long_list_heading": long_list_head_fmt,
    }
