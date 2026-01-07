import csv
import os
from datetime import datetime
from icalendar import Calendar


def to_datetime(dt):
    """Convert iCalendar datetime to datetime"""
    if isinstance(dt, datetime):
        return dt
    return datetime.combine(dt, datetime.min.time())


def export():
    print('\nWhich file to export?')
    files = []
    for file in os.listdir("input/"):
        if file.endswith(".ics"):
            print(f" {str(len(files))} --- {file}")
            files.append(file)

    selection = int(input("Select file to use: "))

    eventName = input("\nEnter name of event to export to .csv: ")

    with open(f"input/{files[selection]}", "rb") as fileReader:
        calendar = Calendar.from_ical(fileReader.read())

    rows = []

    for component in calendar.walk():
        if component.name != "VEVENT":
            continue

        summary = str(component.get("summary", ""))
        if eventName.lower() not in summary.lower():
            continue

        start = to_datetime(component.get("dtstart").dt)
        end = to_datetime(component.get("dtend").dt)
        description = str(component.get("description", ""))

        rows.append([summary, start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), description.replace("\n", " ")])

        rows.sort(key=lambda x: x[1])

    try:
        with open(f"output/{files[selection]}.csv", "w", newline="", encoding="utf-8") as fileWriter:
            writer = csv.writer(fileWriter)
            writer.writerow(["Summary", "Start Time", "End Time", "Description"])
            writer.writerows(rows)

        print(f"Saved {len(rows)} matching events to output/{files[selection]}.csv")
    except PermissionError:
        print("\nPermission denied, File is not accessible.")
