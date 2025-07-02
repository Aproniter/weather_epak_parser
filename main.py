import csv
from dataclasses import fields
import os
import asyncio
from datetime import datetime
import pytz
import redis.asyncio as aioredis
from concurrent.futures import ProcessPoolExecutor

from reader import get_data
from schemas import Result, WeatherData


pool = aioredis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=20)
redis_client = aioredis.Redis(connection_pool=pool)

executor = ProcessPoolExecutor(max_workers=4)

def process_location(weather_data, loc):
    name, lat, long = loc
    field_names = [f.name for f in fields(WeatherData) if f.name != 'unit_descriptors']
    results = []
    local_tz = pytz.timezone('Europe/Moscow')
    for field_name in field_names:
        field = getattr(weather_data, field_name)
        unit = weather_data.unit_descriptors[field_name]
        value = field.field().bilinear(long, lat)
        time_field = pytz.utc.localize(datetime.strptime(field.valid_time(), "%Y-%m-%dT%H:%MZ")).astimezone(local_tz).strftime('%Y-%m-%dT%H:%MZ')
        formatted = unit.format(value)['formattedVal']
        results.append(formatted)
        results.append(time_field)

    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return Result(name, dt, *results)

async def handle_data(weather_data: WeatherData, locations):
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            executor,
            process_location,
            weather_data,
            loc
        )
        for loc in locations
    ]
    results = await asyncio.gather(*tasks)
    return results

async def process_locations(locations=None, file_locations_path=None):
    weather_data: WeatherData = await get_data(redis_client)

    if locations is None and file_locations_path is None:
        raise ValueError('No locations to process')

    if locations is None and file_locations_path:
        locations = []
        with open(file_locations_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            _ = next(reader)
            for row in reader:
                label = row[0]
                latitude = float(row[1])
                longitude = float(row[2])
                locations.append([label, latitude, longitude])
    results = await handle_data(weather_data, locations)
    
    results_dict = {}
    for result in results:
        loc = next((loc for loc in locations if loc[0] == result.name), None)
        if loc:
            key = (loc[1], loc[2])  # (lat, lon)
            results_dict[key] = result.__dict__
    return results_dict

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations.csv')
    result = asyncio.run(process_locations(file_locations_path=csv_path))
    for coord, data in result.items():
        print(coord, data)