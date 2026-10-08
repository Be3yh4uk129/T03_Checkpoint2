def close_count(k, ref):
    """Iteration counts may differ by a few iterations from round-off."""
    return abs(k - ref) <= max(5, 0.002 * ref)
