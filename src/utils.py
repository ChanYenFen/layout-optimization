def remap(value, origin_min, origin_max, final_min, final_max):
    if origin_min == origin_max:
        raise ValueError("origin_min and origin_max must be different.")

    ratio = (value - origin_min) / (origin_max - origin_min)
    return final_min + ratio * (final_max - final_min)
