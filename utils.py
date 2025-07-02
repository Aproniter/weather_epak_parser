import hashlib
import math
import time
import numpy as np
from datetime import datetime, timezone

from config import redis_expire_seconds


def make_cache_key(key_prefix, *args):
    return 'cache:' + key_prefix + str(args)

async def cache_binary_data(redis_client, data: bytes, *args):
    key = make_cache_key('my_binary_data', *args)
    await redis_client.set(key, data, ex=redis_expire_seconds)

async def get_cached_binary_data(redis_client, *args):
    key = make_cache_key('my_binary_data', *args)
    cached = await redis_client.get(key)
    if cached is not None:
        print('Cache hit')
        return cached
    print('Cache miss')
    return None

def decimalize(x):
    return float(x)

def floor_mod(x, n):
    return ((x % n) + n) % n

def bilinear_scalar(grid, data):
    def bilinear(lam, phi):
        indices = grid.closest4(lam, phi)

        i00 = indices[0]
        if not (isinstance(i00, float) and math.isnan(i00)):
            i10, i01, i11 = indices[1], indices[2], indices[3]
            x, y = indices[4], indices[5]
            rx, ry = 1 - x, 1 - y

            v00 = data[int(i00)] if not math.isnan(i00) else float('nan')
            v10 = data[int(i10)] if not (isinstance(i10, float) and math.isnan(i10)) else float('nan')
            v01 = data[int(i01)] if not (isinstance(i01, float) and math.isnan(i01)) else float('nan')
            v11 = data[int(i11)] if not (isinstance(i11, float) and math.isnan(i11)) else float('nan')

            def is_valid(v):
                return not (isinstance(v, float) and math.isnan(v))

            if is_valid(v00):
                if is_valid(v10) and is_valid(v01) and is_valid(v11):
                    a = rx * ry
                    b = x * ry
                    c = rx * y
                    d = x * y
                    return v00 * a + v10 * b + v01 * c + v11 * d

                elif is_valid(v11) and is_valid(v10) and x >= y:
                    return v10 + rx * (v00 - v10) + y * (v11 - v10)

                elif is_valid(v01) and is_valid(v11) and x < y:
                    return v01 + x * (v11 - v01) + ry * (v00 - v01)

                elif is_valid(v01) and is_valid(v10) and x <= ry:
                    return v00 + x * (v10 - v00) + y * (v01 - v00)

            elif is_valid(v11) and is_valid(v01) and is_valid(v10) and x > ry:
                return v11 + rx * (v01 - v11) + ry * (v10 - v11)

        return float('nan')

    bilinear.webgl = None
    return bilinear

def legacy_blockify(epak, selector):
    variables = epak['header']['variables']
    blocks = []

    for key in list(variables.keys()):
        if selector.search(key):
            v = variables[key]
            if isinstance(v['data'], dict) and 'block' in v['data']:
                continue

            blocks.append(np.array(v['data'], dtype=np.float32))
            v['data'] = {'block': len(blocks) - 1}

    return {
        'header': epak,
        'blocks': blocks,
    }

def legacy_munge(records, var_names, dimensions=None):
    header = records[0]['header']

    from datetime import timedelta

    def utc_add(dt, **kwargs):
        hours = kwargs.get('hour', 0)
        return dt + timedelta(hours=hours)

    valid_time = utc_add(header['refTime'], hour=header['forecastTime'])

    variables = {
        'time': {'data': [valid_time.isoformat()]},
        'lat': {'sequence': {'start': header['la1'], 'delta': -header['dy'], 'size': header['ny']}},
        'lon': {'sequence': {'start': header['lo1'], 'delta': header['dx'], 'size': header['nx']}},
    }

    blocks = []

    for i, key in enumerate(var_names):
        variables[key] = {
            'dimensions': dimensions or ['time', 'lat', 'lon'],
            'data': {'block': i}
        }
        blocks.append(np.array(records[i]['data'], dtype=np.float32))


    return {
        'header': {'variables': variables},
        'blocks': blocks,
    }

def nearest_scalar(grid, data):
    def nearest(lam, phi):
        idx = grid.closest(lam, phi)
        if isinstance(idx, float) and math.isnan(idx):
            return float('nan')
        try:
            return data[int(idx)]
        except IndexError:
            return float('nan')
    nearest.webgl = None
    return nearest

def as_date(dt):
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    elif isinstance(dt, str):
        return datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)
    else:
        raise TypeError('Unsupported date type')

def get_date_time_string(dt):
    date_str = dt.strftime('%Y/%m/%d')
    time_str = dt.strftime('%H%M')
    return date_str, time_str

def celsium(x):
    return x - 273.15

def farengate(x):
    return (x * 9) / 5 - 459.67

def kelvin(x):
    return x

def pascal(x):
    return x / 100

def cloud(x):
    return x
