import math


RAD = math.pi / 180
redis_expire_seconds = 3600

def get_url(selector, datetime_point):
    selectors = {
        'Temperature': 'temp-surface-level-gfs-0.5.epak',
        'Dewpoint': 'dew_point_temp-2m-gfs-0.5.epak',
        'Pressure': 'mean_sea_level_pressure-gfs-0.5.epak',
        'Cloud': 'total_cloud_water-gfs-0.5.epak',
        'Precipitable': 'total_precipitable_water-gfs-0.5.epak'
    }
    return f'https://gaia.nullschool.net/data/gfs/{datetime_point}-{selectors[selector]}'
