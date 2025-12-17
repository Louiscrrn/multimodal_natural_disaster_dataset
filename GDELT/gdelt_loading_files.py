import requests

def load_gdelt_index():
    r = requests.get(
        "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
        timeout=60
    )
    r.raise_for_status()

    filenames = []

    for line in r.text.splitlines():
        parts = line.split()
        url = parts[-1]                    
        filename = url.rsplit("/", 1)[-1]
        filenames.append(filename)

    return filenames


def get_files_for_day_3h(date_str, gdelt_files, type="mentions.CSV"): 
    """ One day has 8 files at 3-hour intervals: """ 
    valid_hours = {"00", "03", "06", "09", "12", "15", "18", "21"} 
    return [ 
        f for f in gdelt_files 
        if f.endswith(f".{type}.zip") 
        and f[:8] == date_str and f[8:10] 
        in valid_hours # heure 
        and f[10:12] == "00" # minutes 
    ]
