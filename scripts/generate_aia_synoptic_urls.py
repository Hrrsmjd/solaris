import datetime

BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic"

# HOURS = ['0000', '0600', '1200', '1800']
HOURS = ["0000", "1200"]
WAVELENGTHS = ["0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600", "1700", "4500"]


def generate_dates(start_year, end_year):
    start_date = (
        datetime.date(2010, 7, 1) if start_year == 2010 else datetime.date(start_year, 1, 1)
    )
    end_date = datetime.date(end_year, 12, 31)
    delta = datetime.timedelta(days=1)
    while start_date <= end_date:
        yield start_date
        start_date += delta


def generate_urls():
    for date in generate_dates(2010, 2023):  ## Change end date to 2018 or later
        for hour in HOURS:
            for wavelength in WAVELENGTHS:
                yield (
                    f"{BASE_URL}/{date.year}/{date.strftime('%m')}/"
                    f"{date.strftime('%d')}/H{hour}/"
                    f"AIA{date.strftime('%Y%m%d')}_{hour[:2]}00_{wavelength}.fits"
                )


# Generate URLs and write to a file
with open("aia_synoptic_urls.txt", "w") as f:
    for url in generate_urls():
        f.write(f"{url}\n")

print("URL list has been written to 'aia_synoptic_urls.txt'")
