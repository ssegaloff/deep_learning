from pathlib import Path
import subprocess
import geopandas as gpd

# config and constants
INPUT_PATH = Path.home() / 'gis-scratch' / 'Virginia.geojson'
INTERMEDIATE_PATH = Path(__file__).parent / 'intermediate' / 'albemarle-buildings-clip.geojson'
OUTPUT_PATH = Path(__file__).parent.parent / 'geo-data-viewer' / 'public' / 'albemarle-buildings.geojson'
BBOX = {'min_lat' : 37.7225831, 'max_lat' : 38.2779351, 'min_lon' : -78.8388741, 'max_lon' : -78.2078860} # coords for Albemarle bounding box from Nominatim

# clipping

# pull the 4 bounding box values out of BBOX by dict key, convert floats to strings, assign to xmin, ymin, xmax, ymax
xmin, ymin, xmax, ymax = str(BBOX['min_lon']), str(BBOX['min_lat']), str(BBOX['max_lon']), str(BBOX['max_lat'])

# create the intermediate clipped file directory if needed
INTERMEDIATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# GDAL's GeoJSON driver refuses to overwrite an existing file, so remove any
# stale output from a previous run before we ask ogr2ogr to create a fresh one.
# (keeps the script rerunnable without manual cleanup)
INTERMEDIATE_PATH.unlink(missing_ok=True)

subprocess.run([ # launches an external process (not python code); pass this a list of strings (the command and arguments)
   'ogr2ogr', # the program being launched (the GDAL command line tool for geographic clipping)
   '-clipsrc', # flag telling ogr2ogr that the next 4 values are a bounding box to clip to
   xmin, ymin, xmax, ymax, # in order requrired by ogr2ogr -clipsrc
   str(INTERMEDIATE_PATH), # tells ogr2ogr where to write the clipped file
   str(INPUT_PATH),
   ], check=True) # raise CalledProcessError if the command fails

print(f'Clipped features written to {INTERMEDIATE_PATH}')

# assign ids and write final output
gdf = gpd.read_file(INTERMEDIATE_PATH)

n_in = len(gdf)
gdf['footprint_id'] = range(n_in)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.unlink(missing_ok=True)  # same overwrite restriction as ogr2ogr — see note above

gdf.to_file(OUTPUT_PATH, driver='GeoJSON')

n_out = len(gdf)
print(f'{n_in} features read from {INTERMEDIATE_PATH}')
print(f'{n_out} features written to {OUTPUT_PATH}')