# Reading and writing the env's variation values as plain data.
#
# A recorded demo stores the fully resolved variation values (including ones
# sampled at random from the seed) as a flat JSON-able dict; replay feeds them
# back through reset(options={'variation': [], 'variation_values': ...}) to
# rebuild the exact same env.
import numpy as np

from pusht_scenarios.env import utils


def cast_values(env, values: dict) -> dict:
    """Cast plain values (YAML/JSON lists and numbers) to each space's dtype and
    shape, as reset(options={'variation_values': ...}) requires."""
    space = env.unwrapped.variation_space
    out = {}
    for key, value in values.items():
        try:
            sub = utils.get_in(space, key.split('.'))
        except (KeyError, TypeError):
            raise ValueError(f'unknown variation {key!r}') from None
        cast = np.asarray(value, dtype=sub.dtype).reshape(sub.shape)
        out[key] = cast[()] if cast.shape == () else cast
    return out


def flatten(env) -> dict:
    """Every variation value right now, as {'block.shape': 2, 'sim.damping': 0.0, ...}
    with plain Python numbers/lists (JSON-safe, exact for floats)."""
    space = env.unwrapped.variation_space
    return {
        name: np.asarray(utils.get_in(space, name.split('.')).value).tolist()
        for name in space.names()
    }


def exact_reset_options(env, values: dict) -> dict:
    """Reset options that rebuild an env with exactly these variation values
    (nothing resampled)."""
    return {'variation': [], 'variation_values': cast_values(env, values)}
