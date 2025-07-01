import asyncio
import re
import aiohttp
from decoder import decode_epak, scalar_product
from utils import cache_binary_data_example, get_cached_binary_data


async def read_temp(redis_client):
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-temp-surface-level-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-temp-surface-level-gfs-0.5.epak'
    print(url)

    cached_data = await get_cached_binary_data(redis_client, param1='temp', param2=123)
    if cached_data is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                epak_data = await response.read()
        await cache_binary_data_example(redis_client, epak_data, param1='temp', param2=123) 
        cached_data = epak_data

    data = decode_epak(cached_data)
    temp = scalar_product(data, re.compile('Temperature'), {
        'hasMissing': False,
        'legacyName': 'Temperature'
    })

    return temp

async def read_dew(redis_client):
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-dew_point_temp-2m-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-dew_point_temp-2m-gfs-0.5.epak'
    print(url)

    cached_data = await get_cached_binary_data(redis_client, param1='dew', param2=123)
    if cached_data is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                epak_data = await response.read()
        await cache_binary_data_example(redis_client, epak_data, param1='dew', param2=123) 
        cached_data = epak_data

    data = decode_epak(cached_data)
    dew = scalar_product(data, re.compile('Dewpoint'), {
        'hasMissing': False,
        'legacyName': 'Dewpoint'
    })

    return dew

async def read_pressure(redis_client):
    # url = f'https://gaia.nullschool.net/data/gfs/{date}/{time}-mean_sea_level_pressure-gfs-0.5.epak'
    url = f'https://gaia.nullschool.net/data/gfs/current/current-mean_sea_level_pressure-gfs-0.5.epak'
    print(url)

    cached_data = await get_cached_binary_data(redis_client, param1='pressure', param2=123)
    if cached_data is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                epak_data = await response.read()
        await cache_binary_data_example(redis_client, epak_data, param1='pressure', param2=123) 
        cached_data = epak_data

    data = decode_epak(cached_data)
    pressure = scalar_product(data, re.compile('Pressure'), {
        'hasMissing': False,
        'legacyName': 'Pressure'
    })

    return pressure

async def get_data(redis_client):
    tasks = [
        read_temp(redis_client),
        read_pressure(redis_client),
        read_dew(redis_client),
    ]
    temp, pressure, dew = await asyncio.gather(*tasks)
    return {
        'temp': temp,
        'pressure': pressure,
        'dew': dew
    }
