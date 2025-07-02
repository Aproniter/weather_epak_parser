import asyncio
import re
import aiohttp
import pytz
from datetime import datetime, timedelta
from decoder import decode_epak, scalar_product
from schemas import WeatherData
from utils import cache_binary_data, get_cached_binary_data
from config import get_url


async def data_with_selector(redis_client, session, scalar_selector, datetime_point='current/current'):
    url = get_url(scalar_selector, datetime_point)
    print(url)
    cached_data = await get_cached_binary_data(redis_client, scalar_selector, datetime_point)
    if cached_data is None:
        async with session.get(url) as response:
            epak_data = await response.read()
        await cache_binary_data(redis_client, epak_data, scalar_selector, datetime_point) 
        cached_data = epak_data
    data = decode_epak(cached_data)
    return scalar_product(data, re.compile(scalar_selector), {
        'hasMissing': False,
        'legacyName': scalar_selector
    })

async def get_data(redis_client):
    local_tz = pytz.timezone('Europe/Moscow')
    local_now = datetime.now(local_tz)
    hour_ahead = local_now + timedelta(hours=1)
    utc_ahead = hour_ahead.astimezone(pytz.utc).replace(minute=0, second=0, microsecond=0)
    datetimepoint = utc_ahead.strftime('%Y/%m/%d/%H%M')
    async with aiohttp.ClientSession() as session:
        tasks = [
            data_with_selector(redis_client, session, selector)
            for selector in ('Temperature', 'Pressure', 'Dewpoint', 'Cloud', 'Precipitable')
        ] + [
            data_with_selector(redis_client, session, selector, datetimepoint)
            for selector in ('Temperature', 'Pressure', 'Dewpoint', 'Cloud', 'Precipitable')
        ]
        weather_data = WeatherData(*await asyncio.gather(*tasks))
    return weather_data
