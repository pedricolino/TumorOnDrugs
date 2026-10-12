"""Styling for positive/negative score cells."""


def color_cells(value):
    """Return CSS for coloring a score cell in a positive/negative table."""
    if value > 0:
        return f"background-color: rgba(0, 0, 255, {min(value, 1) * 3})"
    return f"background-color: rgba(255, 0, 0, {min(abs(value), 1) * 3})"
