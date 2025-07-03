import asyncio
import re
import aiohttp
import pytz
from datetime import datetime, timedelta
from decoder import decode_epak, scalar_product
from schemas import WeatherData
from utils import cache_binary_data, get_cached_binary_data
from config import Config, get_url


config = Config()

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

def format_datetimepoint(local_now: datetime, hours_offset: int) -> str:
    hour_ahead = local_now + timedelta(hours=hours_offset)
    utc_ahead = hour_ahead.astimezone(pytz.utc).replace(minute=0, second=0, microsecond=0)
    return utc_ahead.strftime('%Y/%m/%d/%H%M')

async def get_data(redis_client):
    local_tz = pytz.timezone('Europe/Moscow')
    local_now = datetime.now(local_tz)
    selected_selectors = config.get("urls.selected", [])
    hours_offsets = config.get("time_offsets.hours", [])
    async with aiohttp.ClientSession() as session:
        tasks = []
        for selector in selected_selectors:
            tasks.append(data_with_selector(redis_client, session, selector))
        for hours in hours_offsets:
            datetimepoint = format_datetimepoint(local_now, hours)
            for selector in selected_selectors:
                tasks.append(data_with_selector(redis_client, session, selector, datetimepoint))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        current_count = len(selected_selectors)
        current_results = results[:current_count]
        ahead_results = results[current_count:]
        ahead_data = {hours: {} for hours in hours_offsets}
        index = 0
        for hours in hours_offsets:
            for selector in selected_selectors:
                ahead_data[hours][selector] = ahead_results[index]
                index += 1
        def get_field(selector):
            try:
                return current_results[selected_selectors.index(selector)]
            except Exception:
                return None

        def get_ahead_field(selector):
            return {h: ahead_data[h].get(selector) for h in hours_offsets}

        weather_data = WeatherData(
            temperature=get_field('Temperature'),
            pressure=get_field('Pressure'),
            dewpoint=get_field('Dewpoint'),
            cloud=get_field('Cloud'),
            precipitable=get_field('Precipitable'),
            temperature_ahead=get_ahead_field('Temperature'),
            pressure_ahead=get_ahead_field('Pressure'),
            dewpoint_ahead=get_ahead_field('Dewpoint'),
            cloud_ahead=get_ahead_field('Cloud'),
            precipitable_ahead=get_ahead_field('Precipitable'),
        )
    return weather_data
