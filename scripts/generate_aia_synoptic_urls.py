import argparse
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic"
HOURS = ("0000", "1200")
WAVELENGTHS = ("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600", "1700", "4500")


def generate_dates(start_year: int, end_year: int):
    start_date = date(2010, 7, 1) if start_year == 2010 else date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    delta = timedelta(days=1)
    while start_date <= end_date:
        yield start_date
        start_date += delta


def generate_urls(start_year: int = 2010, end_year: int = 2023):
    for current_date in generate_dates(start_year, end_year):
        for hour in HOURS:
            for wavelength in WAVELENGTHS:
                yield (
                    f"{BASE_URL}/{current_date.year}/{current_date:%m}/"
                    f"{current_date:%d}/H{hour}/"
                    f"AIA{current_date:%Y%m%d}_{hour[:2]}00_{wavelength}.fits"
                )


def main():
    parser = argparse.ArgumentParser(description="Generate JSOC synoptic FITS URLs.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=Path("aia_synoptic_urls.txt"))
    args = parser.parse_args()

    with args.output.open("w", encoding="utf-8") as file:
        for url in generate_urls(args.start_year, args.end_year):
            file.write(f"{url}\n")

    print(f"URL list has been written to '{args.output}'")


if __name__ == "__main__":
    main()
