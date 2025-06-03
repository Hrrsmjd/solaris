from datetime import datetime,timedelta

def parse_custom_hour(hour_str):
    """Convert custom hour format HHHH into an integer hour."""
    return int(hour_str[1:3])

def to_custom_hour(hour):
    """Convert an integer hour into custom time format HHHH."""
    return f'H{hour:02d}00'

def add_hours(date_time_list, hours_to_add):
    year, month, day, hour_str = date_time_list
    hour = parse_custom_hour(hour_str)
    
    original_datetime = datetime(int(year), int(month), int(day), hour)
    
    new_datetime = original_datetime + timedelta(hours=hours_to_add)
    
    new_hour_str = to_custom_hour(new_datetime.hour)
    
    return [str(new_datetime.year), f"{new_datetime.month:02d}", f"{new_datetime.day:02d}", new_hour_str]
