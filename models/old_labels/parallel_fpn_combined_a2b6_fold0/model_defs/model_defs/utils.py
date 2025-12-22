import math


def raw_growth_fn(level, F_I, G_R):
    return int(F_I * (G_R ** level))  
    # return FEATURE_PROGRESSION[level-1]

def power_2_rounding(raw_num):
    sqrt_val = math.sqrt(raw_num)
    rounding_factor = 2 ** int(math.log2(sqrt_val))
    return int(((raw_num + rounding_factor - 1) // rounding_factor) * rounding_factor)

def growth_fn(level, F_I, G_R):
    #rounds to floor power of 2 (log2(raw))
    #raw=15 → sqrt=3.87 → log2=1.95 → floor=1 → round to 2^1=2 → 16
    #raw 256 => sqrt 16 log2 = > 4 floor round to nearest 2^4 = 16
    return power_2_rounding(raw_growth_fn(level, F_I, G_R))

def dc_growth_fn(level, F_I, G_R, DC):
    return power_2_rounding(raw_growth_fn(level, F_I, G_R)/DC)