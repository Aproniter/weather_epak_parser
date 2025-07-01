import csv
import os
import asyncio
from datetime import datetime
import redis.asyncio as aioredis
from concurrent.futures import ProcessPoolExecutor

from decoder import celsium, farengate, kelvin, pascal
from reader import get_data
from schemas import UnitDescriptor


redis_client = aioredis.Redis(host='localhost', port=6379, db=0)

executor = ProcessPoolExecutor(max_workers=4)

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

async def process_locations(locations=None, file_locations_path=None):
    data = await get_data(redis_client)
    temp = data['temp']
    pressure = data['pressure']
    dew = data['dew']

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
        
    results = await handle_data(temp, pressure, dew, locations, unit_descriptors)
    
    results_dict = {}
    for name, dt, temp_f, dew_f, pressure_f in results:
        loc = next((loc for loc in locations if loc[0] == name), None)
        if loc:
            key = (loc[1], loc[2])  # (lat, lon)
            results_dict[key] = {
                'name': name,
                'datetime': dt,
                'temp': temp_f,
                'dew': dew_f,
                'pressure': pressure_f
            }
    return results_dict

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations.csv')
    result = asyncio.run(process_locations(file_locations_path=csv_path))
    for coord, data in result.items():
        print(coord, data)