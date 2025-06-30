import csv
import os
import asyncio
from datetime import datetime, timedelta, timezone
import json
import math
import re
import struct
import aiohttp
import numpy as np
from concurrent.futures import ProcessPoolExecutor


executor = ProcessPoolExecutor(max_workers=4)

class Field:
    def __init__(self, grid, data):
        self.grid = grid
        self.data = data

    def value_at(self, i):
        return self.data[i]

    def scalarize(self, x):
        return x

    def is_defined(self, i):
        return not np.isnan(self.data[i])

    @property
    def nearest(self):
        return nearest_scalar(self.grid, self.data)

    @property
    def bilinear(self):
        return bilinear_scalar(self.grid, self.data)

class Result:
    def __init__(self, time, field, grid):
        self._time = time
        self._field = field
        self._grid = grid

    def valid_time(self):
        return parts(self._time['data'][0])

    def grid(self):
        return self._grid

    def field(self):
        return self._field


RAD = math.pi / 180

def decimalize(x):
    return float(x)

def floor_mod(x, n):
    return ((x % n) + n) % n

class RegularGrid:
    def __init__(self, lambda_axis, phi_axis):
        self.nx = int(math.floor(lambda_axis['size']))
        self.ny = int(math.floor(phi_axis['size']))
        self.np = self.nx * self.ny

        self.Dlambda = decimalize(lambda_axis['delta'])
        self.Dphi = decimalize(phi_axis['delta'])
        self.lambda0 = decimalize(lambda_axis['start'])
        self.phi0 = decimalize(phi_axis['start'])

        self.lambda1 = self.lambda0 + self.Dlambda * (self.nx - 1)
        self.phi1 = self.phi0 + self.Dphi * (self.ny - 1)

        self.lambda_low = (self.lambda0 - self.Dlambda / 2) * RAD
        self.lambda_high = (self.lambda1 + self.Dlambda / 2) * RAD
        self.lambda_size = self.lambda_high - self.lambda_low

        self.phi_low = (self.phi0 - self.Dphi / 2) * RAD
        self.phi_high = (self.phi1 + self.Dphi / 2) * RAD
        self.phi_size = self.phi_high - self.phi_low

        self.low = [self.lambda_low, self.phi_low]
        self.size = [self.lambda_size, self.phi_size]

        self.is_cylindrical = (math.floor(self.nx * self.Dlambda) >= 360)

    def dimensions(self):
        return {'width': self.nx, 'height': self.ny}

    def is_cylindrical(self):
        return self.is_cylindrical

    def for_each(self, cb, start=0):
        for i in range(start, self.np):
            x = i % self.nx
            y = i // self.nx
            lam = self.lambda0 + x * self.Dlambda
            phi = self.phi0 + y * self.Dphi
            if cb(lam, phi, i):
                return i + 1
        return float('nan')

    def closest(self, lam, phi):
        if not (math.isnan(lam) or math.isnan(phi)):
            x = floor_mod(lam - self.lambda0, 360) / self.Dlambda
            y = (phi - self.phi0) / self.Dphi
            rx = round(x)
            ry = round(y)
            if 0 <= ry < self.ny and 0 <= rx < self.nx or (rx == self.nx and self.is_cylindrical):
                i = ry * self.nx + rx
                if rx == self.nx:
                    return i - self.nx
                return i
        return float('nan')

    def closest4(self, lam, phi):
        if not (math.isnan(lam) or math.isnan(phi)):
            x = floor_mod(lam - self.lambda0, 360) / self.Dlambda
            y = (phi - self.phi0) / self.Dphi
            fx = math.floor(x)
            fy = math.floor(y)
            cx = fx + 1
            cy = fy + 1
            dx = x - fx
            dy = y - fy

            if 0 <= fy < self.ny and 0 <= fx < self.nx and cy < self.ny and (cx < self.nx or (cx == self.nx and self.is_cylindrical)):
                i00 = fy * self.nx + fx
                i10 = i00 + 1
                i01 = i00 + self.nx
                i11 = i01 + 1
                if cx == self.nx:
                    i10 -= self.nx
                    i11 -= self.nx
                return [i00, i10, i01, i11, dx, dy]
        return [float('nan')] * 6

    def webgl(self):
        REGULAR_FRAG = "...shader source..."
        return {
            'shaderSource': REGULAR_FRAG,
            'uniforms': {'u_Low': self.low, 'u_Size': self.size}
        }

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

def parts(date):
    date = as_date(date)
    return {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'hour': date.hour,
        'minute': date.minute,
        'second': date.second,
        'milli': int(date.microsecond / 1000),
    }


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

    grid = RegularGrid(lon['sequence'], lat['sequence'])
    field_instance = Field(grid, data)
    return Result(time, field_instance, grid)

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

async def read_temp():
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-temp-surface-level-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-temp-surface-level-gfs-0.5.epak'
    print(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            epak_data = await response.read()

    data = decode_epak(epak_data)
    temp = scalar_product(data, re.compile('Temperature'), {
        'hasMissing': False,
        'legacyName': 'Temperature'
    })

    return temp

async def read_dew():
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-dew_point_temp-2m-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-dew_point_temp-2m-gfs-0.5.epak'
    print(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            epak_data = await response.read()

    data = decode_epak(epak_data)
    dew = scalar_product(data, re.compile('Dewpoint'), {
        'hasMissing': False,
        'legacyName': 'Dewpoint'
    })

    return dew

async def read_pressure():
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-mean_sea_level_pressure-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-mean_sea_level_pressure-gfs-0.5.epak'
    print(url)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            epak_data = await response.read()

    data = decode_epak(epak_data)
    pressure = scalar_product(data, re.compile('Pressure'), {
        'hasMissing': False,
        'legacyName': 'Pressure'
    })

    return pressure

def get_date_time_string(dt):
    date_str = dt.strftime('%Y/%m/%d')
    time_str = dt.strftime('%H%M')
    return date_str, time_str

async def get_data():
    tasks = [
        read_temp(),
        read_pressure(),
        read_dew(),
    ]
    temp, pressure, dew = await asyncio.gather(*tasks)
    return {
        'temp': temp,
        'pressure': pressure,
        'dew': dew
    }


class UnitDescriptor:
    def __init__(self, convert_func, precision=1, symbol=''):
        self.convert_func = convert_func
        self.precision = precision
        self.symbol = symbol

    def convert(self, x):
        return self.convert_func(x)

    def format(self, x):
        val = self.convert(x)
        formatted_val = f"{val:.{self.precision}f} {self.symbol}"
        return {'formattedVal': formatted_val}
    
def celsium(x):
    return x - 273.15

def farengate(x):
    return (x * 9) / 5 - 459.67

def kelvin(x):
    return x

def pascal(x):
    return x / 100

unit_descriptors = {
    '°C': UnitDescriptor(celsium, precision=1, symbol='°C'),
    '°F': UnitDescriptor(farengate, precision=1, symbol='°F'),
    'K': UnitDescriptor(kelvin, precision=1, symbol='K'),
    'hPa': UnitDescriptor(pascal, precision=1, symbol='hPa'),
}

def process_location(temp, pressure, dew, loc, temp_unit_descriptors):
    name, lat, long = loc

    temp_val = temp.field().bilinear(long, lat)
    temp_formatted = temp_unit_descriptors['°C'].format(temp_val)['formattedVal']

    pressure_val = pressure.field().bilinear(long, lat)
    pressure_formatted = temp_unit_descriptors['hPa'].format(pressure_val)['formattedVal']

    dew_val = dew.field().bilinear(long, lat)
    dew_formatted = temp_unit_descriptors['°C'].format(dew_val)['formattedVal']

    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return (name, dt, temp_formatted, dew_formatted, pressure_formatted)

async def handle_data(temp, pressure, dew, locations, temp_unit_descriptors):
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            executor,
            process_location,
            temp,
            pressure,
            dew,
            loc,
            temp_unit_descriptors
        )
        for loc in locations
    ]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    locations = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations.csv')
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)
        for row in reader:
            label = row[0]
            latitude = float(row[1])
            longitude = float(row[2])
            locations.append([label, latitude, longitude])

    data = await get_data()

    temp = data['temp']
    pressure = data['pressure']
    dew = data['dew']

    results = await handle_data(temp, pressure, dew, locations, unit_descriptors)

    for name, dt, temp_f, dew_f, pressure_f in results:
        print(name, dt, temp_f, dew_f, pressure_f)

if __name__ == '__main__':
    asyncio.run(main())
