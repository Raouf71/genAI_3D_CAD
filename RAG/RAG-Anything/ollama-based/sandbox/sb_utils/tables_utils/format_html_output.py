import re
import html

COLUMN_LABELS = [
    "Z2",
    "ZB[mm]",
    "0B[mm]",
    "OTK[mm]",
    "OKK[mm]",
    "ON[mm]",
    "L[mm]",
    "OFM[mm]",
    "Ws[mm]",
    "G[g]",
    "DM**[Ncm]",
    "Art-Nr.",
]

TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)

def format_gear_row(row_html: str) -> str:
    # Extract cell contents
    raw_cells = TD_PATTERN.findall(row_html)
    cells = [html.unescape(c).strip() for c in raw_cells]

    if not cells:
        return "PARSE_ERROR: no <td> cells found"

    # Build header
    z2 = cells[0]
    lines = [f"#### Dimensions for Z2 = {z2} Teeth Spur Gear"]

    # Map labels to values (truncate to shorter length to stay safe)
    for label, value in zip(COLUMN_LABELS, cells):
        lines.append(f"* {label}: {value}")

    return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    row = """<tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>3375</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>11,39</td><td rowspan=1 colspan=1>61,36</td><td rowspan=1 colspan=1>SH12525HF</td></tr>"""
    print(format_gear_row(row))
