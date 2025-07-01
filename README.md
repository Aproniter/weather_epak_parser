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

3. Results will be printed to the console with coordinates and values for 
temperature, dew point, and pressure.

## Project Structure 

- `main.py` — main data processing script. 
- `decoder.py` — data decoding functions. 
- `reader.py` — data loading. 
- `schemas.py` — unit definitions and formatting. 

## License 

This project is licensed under the MIT License. This permissive license allows 
you to freely use, copy, modify, merge, publish, distribute, sublicense, and/o 
sell copies of the software, provided that the original copyright notice and   
this permission notice are included in all copies or substantial portions of t 
software. The software is provided "as is", without warranty of any kind, 
express or implied. 

For more details, see the full text of the MIT License: 
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) 