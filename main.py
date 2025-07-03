import csv
from datetime import datetime
import os
import asyncio
import pytz
import redis.asyncio as aioredis
from concurrent.futures import ProcessPoolExecutor

from config import Config
from reader import get_data
from schemas import Result, WeatherData


executor = ProcessPoolExecutor(max_workers=4)

def parsing_location(weather_data, loc):
    config = Config()
    name, lat, long = loc
    local_tz = pytz.timezone(config.get("local.timezone", "UTC"))

    def extract_val_time(value_obj, unit):
        val = value_obj.field().bilinear(long, lat)
        time_field = pytz.utc.localize(datetime.strptime(value_obj.valid_time(), "%Y-%m-%dT%H:%MZ")).astimezone(local_tz).strftime('%Y-%m-%dT%H:%MZ')
        formatted = unit.format(val)['formattedVal']
        return formatted, time_field
    
    def extract_ahead(ahead_dict, unit):
        if not ahead_dict:
            return {}
        return {
            offset: extract_val_time(val_obj, unit)
            for offset, val_obj in ahead_dict.items()
        }
    
    fields_info = tuple(map(lambda x: x.lower() ,config.get("urls.selected")))
    
    simple_results = {
        f"{f_in}_f": extract_val_time(getattr(weather_data, f_in), weather_data.unit_descriptors[f_in])
        for f_in in fields_info
    }

    ahead_results = {
        f"{f_in}_ahead_f": extract_ahead(getattr(weather_data, f"{f_in}_ahead"), weather_data.unit_descriptors[f_in])
        for f_in in fields_info
    }

    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result_kwargs = {'name': name, 'dt': dt}

    for out_field_name, (val, time_val) in simple_results.items():
        result_kwargs[out_field_name] = val
        result_kwargs[out_field_name + '_time'] = time_val

    result_kwargs.update(ahead_results)

    return Result(**result_kwargs)

async def handle_data(weather_data: WeatherData, locations, executor):
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            executor,
            parsing_location,
            weather_data,
            loc
        )
        for loc in locations
    ]
    try:
        results = await asyncio.gather(*tasks)
    except AttributeError as e:
        if "'ConnectionError' object has no attribute 'field'" in str(e):
            print("Redis not found")
        else:
            raise AttributeError(e)
    return results

async def process_locations(locations=None, file_locations_path=None, config_params: dict={}):
    config = Config()
    config.init(config_params)
    redis_conf = config.get("redis", {})
    pool = aioredis.ConnectionPool(
        host=redis_conf.get("host", "localhost"),
        port=redis_conf.get("port", 6379),
        db=redis_conf.get("db", 0),
        max_connections=redis_conf.get("max_connections", 20)
    )
    redis_client = aioredis.Redis(connection_pool=pool)
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
    results = await handle_data(weather_data, locations, executor)
    results_dict = {}
    for result in results:
        loc = next((loc for loc in locations if loc[0] == result.name), None)
        if loc:
            key = (loc[1], loc[2])  # (lat, lon)
            results_dict[key] = result.__dict__
    return results_dict

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations.csv')
    result = asyncio.run(process_locations(
        file_locations_path=csv_path,
        config_params={
            'urls': {'selected': ['Temperature']},
            'time_offsets': {'hours': []}
        }
    ))
    for coord, data in result.items():
        print(coord, data)
