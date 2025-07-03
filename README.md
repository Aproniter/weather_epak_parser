# Earth Nullschool .epak Files Parser 

This project is designed to decode and parse `.epak` files obtained from 
the [earth.nullschool.net](https://earth.nullschool.net) website. It is based  
the [earth-nullschool-scraper](https://github.com/suphon-t/earth-nullschool-scrape 
project. 

## Description 

The script downloads data on temperature, pressure, and dew point, decodes it, 
and 
calculates values for specified geographic coordinates. The results are 
formatted 
and displayed using various units of measurement. 

## Features 

- Asynchronous data loading and processing using Redis. 
- Parallel coordinate processing with ProcessPoolExecutor. 
- Reading coordinates from a CSV file. 
- Formatted output of results with timestamps. 

## Installation 

1. Clone the repository: 
    ```bash 
    git clone <repository_URL> 
    cd <project_folder> 
    ``` 

2. Install dependencies:
    Use uv
    ```bash 
    pipx install uv
    uv sync
    ```

3. Ensure Redis server is running on localhost:6379. 

## Usage

1. Prepare a CSV file with locations `locations.csv` in the format: 
    ```
    name,latitude,longitude
    London,51.5074,-0.1278
    Paris,49.8566,2.3522
    ```

2. Run the script:
    ```bash
    uv run main.py
    ```
    By default, the script will:  
    - Read locations from `locations.csv`  
    - Use configuration from `default_config.toml`  
    - Fetch selected weather parameters (e.g., Temperature, Pressure, Dewpoint, etc.)  
    - Use Redis settings specified in the config for caching

3. You can customize the execution by modifying the `main.py` script. For example:

    ```python
    import os
    import asyncio
    from your_module import process_locations  # import the processing function

    if __name__ == '__main__':
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locations.csv')

        result = asyncio.run(process_locations(
            file_locations_path=csv_path,  # path to locations CSV

            # Or specify a list of locations manually:
            # locations=[('Moscow', 55.7558, 37.6173)],

            # Or provide a custom config dictionary:
            # config_params={
            #     'urls': {'selected': ['Temperature']},
            #     'time_offsets': {'hours': [3, 6]}
            # }

            # If omitted, the default_config.toml will be used
        ))

        for coord, data in result.items():
            print(coord, data)
    ```

4. Example `default_config.toml` configuration file:

    ```toml
    [urls]
    base = "https://gaia.nullschool.net/data/gfs"
    selected = ["Temperature", "Pressure", "Dewpoint", "Cloud", "Precipitable"]

    [urls.selectors]
    Temperature = "temp-surface-level-gfs-0.5.epak"
    Dewpoint = "dew_point_temp-2m-gfs-0.5.epak"
    Pressure = "mean_sea_level_pressure-gfs-0.5.epak"
    Cloud = "total_cloud_water-gfs-0.5.epak"
    Precipitable = "total_precipitable_water-gfs-0.5.epak"

    [redis]
    host = "localhost"
    port = 6379
    db = 0
    max_connections = 20
    expire_seconds = 3600

    [local]
    timezone = "Europe/Moscow"

    [time_offsets]
    hours = []
    ```

5. The results will be printed to the console showing the coordinates along with  
   values for temperature, dew point, pressure, and other selected parameters.

## Project Structure

- `main.py` — main data processing script.
- `decoder.py` — data decoding functions.
- `reader.py` — data loading.
- `schemas.py` — unit definitions and formatting.
- `default_config.toml` — configuration file.

## License

This project is licensed under the MIT License. This permissive license allows
you to freely use, copy, modify, merge, publish, distribute, sublicense, and/o
sell copies of the software, provided that the original copyright notice and 
this permission notice are included in all copies or substantial portions of t
software. The software is provided "as is", without warranty of any kind,
express or implied.

For more details, see the full text of the MIT License:
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)
