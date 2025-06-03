## How to download AIA synoptic data

1. Generate URLs using `generate_aia_synoptic_urls.py`.
2. Run `wget -i aia_synoptic_urls.txt -P PATH -x -c`. ## CHANGE PATH
    - The `-x` option preserves the folder structre of the downloaded files.
    - The `-c` option adds the continue/resume capability.
