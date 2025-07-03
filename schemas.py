import math
from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from utils import as_date, bilinear_scalar, decimalize, floor_mod, nearest_scalar, celsium, farengate, kelvin, pascal, cloud


class UnitDescriptor:
    def __init__(self, convert_func, precision=1, symbol=''):
        self.convert_func = convert_func
        self.precision = precision
        self.symbol = symbol

    def convert(self, x):
        return self.convert_func(x)

    def format(self, x):
        val = self.convert(x)
        formatted_val = f'{val:.{self.precision}f} {self.symbol}'
        return {'formattedVal': formatted_val}

unit_descriptors = {
    '°C': UnitDescriptor(celsium, precision=1, symbol='°C'),
    '°F': UnitDescriptor(farengate, precision=1, symbol='°F'),
    'K': UnitDescriptor(kelvin, precision=1, symbol='K'),
    'hPa': UnitDescriptor(pascal, precision=1, symbol='hPa'),
    'kgM': UnitDescriptor(cloud, precision=1, symbol='kgM'),
}

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

class Fields:
    def __init__(self, time, field, grid):
        self._time = time
        self._field = field
        self._grid = grid

    def valid_time(self):
        return tuple(set(self._time['data']))[0]

    def grid(self):
        return self._grid

    def field(self):
        return self._field

@dataclass
class WeatherData:
    temperature: Optional[Fields]
    pressure: Optional[Fields]
    dewpoint: Optional[Fields]
    cloud: Optional[Fields]
    precipitable: Optional[Fields]

    temperature_ahead: Dict[int, Fields] = field(default_factory=dict)
    pressure_ahead: Dict[int, Fields] = field(default_factory=dict)
    dewpoint_ahead: Dict[int, Fields] = field(default_factory=dict)
    cloud_ahead: Dict[int, Fields] = field(default_factory=dict)
    precipitable_ahead: Dict[int, Fields] = field(default_factory=dict)
    unit_descriptors: dict = field(default_factory=dict)

    def __post_init__(self):
        self.unit_descriptors = {
            'temperature': unit_descriptors['°C'],
            'pressure': unit_descriptors['hPa'],
            'dewpoint': unit_descriptors['°C'],
            'cloud': unit_descriptors['kgM'],
            'precipitable': unit_descriptors['kgM'],
            'temperature_ahead': unit_descriptors['°C'],
            'pressure_ahead': unit_descriptors['hPa'],
            'dewpoint_ahead': unit_descriptors['°C'],
            'cloud_ahead': unit_descriptors['kgM'],
            'precipitable_ahead': unit_descriptors['kgM'],
        }

@dataclass
class Result:
    name: str
    dt: str
    temperature_f: Optional[float] = None
    temperature_f_time: Optional[datetime] = None
    dewpoint_f: Optional[float] = None
    dewpoint_f_time: Optional[datetime] = None
    pressure_f: Optional[float] = None
    pressure_f_time: Optional[datetime] = None
    cloud_f: Optional[float] = None
    cloud_f_time: Optional[datetime] = None
    precipitable_f: Optional[float] = None
    precipitable_f_time: Optional[datetime] = None

    temperature_ahead_f: Dict[int, Tuple[float, datetime]] = field(default_factory=dict)
    dewpoint_ahead_f: Dict[int, Tuple[float, datetime]] = field(default_factory=dict)
    pressure_ahead_f: Dict[int, Tuple[float, datetime]] = field(default_factory=dict)
    cloud_ahead_f: Dict[int, Tuple[float, datetime]] = field(default_factory=dict)
    precipitable_ahead_f: Dict[int, Tuple[float, datetime]] = field(default_factory=dict)


class RegularGrid:
    RAD = math.pi / 180

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

        self.lambda_low = (self.lambda0 - self.Dlambda / 2) * self.RAD
        self.lambda_high = (self.lambda1 + self.Dlambda / 2) * self.RAD
        self.lambda_size = self.lambda_high - self.lambda_low

        self.phi_low = (self.phi0 - self.Dphi / 2) * self.RAD
        self.phi_high = (self.phi1 + self.Dphi / 2) * self.RAD
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
