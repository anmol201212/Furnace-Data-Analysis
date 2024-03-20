import csv
import numpy as np

def read_csv(filename):
    data = []
    with open(filename, 'r', newline='',encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def calculate_statistics(data):
    # Skip header row and convert data to numpy array

    numeric_data = np.array(data[1:], dtype=float)
    numeric_data = numeric_data[:,1:5]
    
    # Calculate statistics for each column
    statistics = {
        'Count': np.count_nonzero(~np.isnan(numeric_data), axis=0),
        'Mean': np.nanmean(numeric_data, axis=0),
        'Std': np.nanstd(numeric_data, axis=0),
        'Min': np.nanmin(numeric_data, axis=0),
        '25%': np.nanpercentile(numeric_data, 25, axis=0),
        '50%': np.nanpercentile(numeric_data, 50, axis=0),
        '75%': np.nanpercentile(numeric_data, 75, axis=0),
        'Max': np.nanmax(numeric_data, axis=0)
    }
    return statistics

def read_csv_with_statistics(filename):
    data = read_csv(filename)
    statistics = calculate_statistics(data)
    return data, statistics
