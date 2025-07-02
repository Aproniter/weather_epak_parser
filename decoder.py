import json
import re
import struct
import numpy as np

import schemas
from utils import legacy_munge


def scalar_product(bundle, selector: re.Pattern, options: dict = None):
    if options is None:
        options = {}

    if isinstance(bundle, list):
        bundle = legacy_munge(bundle, [options.get('legacyName')])

    # if not hasattr(bundle, 'blocks'):
    #     bundle = legacy_blockify(bundle, selector)

    epak = bundle

    vars_ = epak['header']['variables']

    x = None
    for key in vars_:
        if selector.search(key):
            x = key
            break
    if x is None:
        raise ValueError('No matching variable found in vars')

    target = vars_[x]
    dims = target['dimensions']

    def last(lst, n):
        return lst[-n:]

    time_dim = dims[0]
    lat_dim, lon_dim = last(dims, 2)

    time = vars_[time_dim]
    lat = vars_[lat_dim]
    lon = vars_[lon_dim]
    data = epak['blocks'][target['data']['block']]

    if 'transform' in options and callable(options['transform']):
        options['transform'](data)

    grid = schemas.RegularGrid(lon['sequence'], lat['sequence'])
    field_instance = schemas.Field(grid, data)
    return schemas.Fields(time, field_instance, grid)

def to_int32(x):
    x = x & 0xFFFFFFFF
    if x & 0x80000000:
        return x - 0x100000000
    return x

def varpack_decode(values, bytes_data):
    i = 0
    j = 0
    length = len(bytes_data)

    while i < length:
        b = bytes_data[i]
        i += 1
        if b < 128:
            b = to_int32(b << 25) >> 25
        else:
            high4 = b >> 4
            if high4 in (0x8, 0x9, 0xa, 0xb):
                part1 = to_int32((b << 26) & 0xFFFFFFFF) >> 18
                part2 = bytes_data[i]
                i += 1
                b = to_int32(part1 | part2)

            elif high4 in (0xc, 0xd):
                part1 = to_int32((b << 27) & 0xFFFFFFFF) >> 11
                part2 = bytes_data[i] << 8
                i += 1
                part3 = bytes_data[i]
                i += 1
                b = to_int32(part1 | part2 | part3)

            elif high4 == 0xe:
                part1 = to_int32((b << 28) & 0xFFFFFFFF) >> 4
                part2 = bytes_data[i] << 16
                i += 1
                part3 = bytes_data[i] << 8
                i += 1
                part4 = bytes_data[i]
                i += 1
                b = to_int32(part1 | part2 | part3 | part4)

            elif high4 == 0xf:
                if b == 255:
                    run = 1 + bytes_data[i]
                    i += 1
                    for _ in range(run):
                        values[j] = float('nan')
                        j += 1
                    continue
                else:
                    subcase = b & 0x07
                    if subcase in (0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7):
                        b = (bytes_data[i] << 24) | (bytes_data[i+1] << 16) | (bytes_data[i+2] << 8) | bytes_data[i+3]
                        i += 4
                        b = to_int32(b)
                    else:
                        raise NotImplementedError("NYI")
            else:
                pass

        values[j] = b
        j += 1

    return values

def undelta_plane(values: np.ndarray, cols: int, rows: int, grids: int):
    for z in range(grids):
        k = z * cols * rows

        for x in range(1, cols):
            i = k + x
            p = values[i - 1]
            values[i] += p if not np.isnan(p) else 0

        for y in range(1, rows):
            j = k + y * cols
            p = values[j - cols]
            values[j] += p if not np.isnan(p) else 0

            for x in range(1, cols):
                i = j + x
                a = values[i - 1]
                b = values[i - cols]
                c = values[i - cols - 1]
                p = a + b - c

                if not np.isnan(p):
                    addition = p
                elif not np.isnan(a):
                    addition = a
                elif not np.isnan(b):
                    addition = b
                elif not np.isnan(c):
                    addition = c
                else:
                    addition = 0

                values[i] += addition
    return values

def dequantize(values, scale_factor):
    values /= scale_factor
    return values

def decode_ppak(bytes_data, cols, rows, grids, scale_factor):
    size = cols * rows * grids
    values = np.empty(size, dtype=np.float32)

    varpack_decode(values, bytes_data)
    undelta_plane(values, cols, rows, grids)
    dequantize(values, scale_factor)

    return values

def decode_ppak_block(block_type, buffer: bytes, offset: int, length: int):
    cols = struct.unpack_from('>I', buffer, offset)[0]
    rows = struct.unpack_from('>I', buffer, offset + 4)[0]
    grids = struct.unpack_from('>I', buffer, offset + 8)[0]
    scale_power = struct.unpack_from('>f', buffer, offset + 12)[0]
    scale_factor = 10 ** scale_power

    bytes_data = buffer[offset + 16 : offset + length]

    values = decode_ppak(bytes_data, cols, rows, grids, scale_factor)

    metadata = {
        'type': block_type,
        'cols': cols,
        'rows': rows,
        'grids': grids,
        'scaleFactor': scale_factor,
    }

    return {
        'metadata': metadata,
        'values': values,
    }


def decode_utf8(data: bytes) -> str:
    return data.decode('utf-8')

def decode_epak(buffer: bytes, header_only=False):
    i = 0
    view = memoryview(buffer)

    head = decode_utf8(view[i:i+4].tobytes())
    i += 4
    if head != "head":
        raise ValueError(f"expected 'head' but found '{head}'")

    length = struct.unpack_from('>I', view, i)[0]
    i += 4

    header_json_bytes = view[i:i+length].tobytes()
    header = json.loads(decode_utf8(header_json_bytes))

    i += length

    blocks = []
    metadata = []

    while not header_only:
        if i + 4 > len(buffer):
            break

        type_str = decode_utf8(view[i:i+4].tobytes())
        if type_str == 'tail':
            break

        i += 4

        if i + 4 > len(buffer):
            raise ValueError('Unexpected end of buffer while reading block length')

        length = struct.unpack_from('>I', view, i)[0]
        i += 4

        if type_str == 'ppak':
            block = decode_ppak_block(type_str, buffer, i, length)
        else:
            raise ValueError(f'unknown block type: {type_str}')

        blocks.append(block['values'])
        metadata.append(block['metadata'])
        i += length

    return {'header': header, 'blocks': blocks, 'metadata': metadata}