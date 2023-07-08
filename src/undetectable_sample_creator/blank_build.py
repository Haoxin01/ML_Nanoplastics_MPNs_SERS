import numpy as np
import pandas as pd

# Load the CSV file and get the x-axis values
input_df = pd.read_csv('Xrange.csv')
x = input_df.iloc[:, 0].values

def generate_parabolas(peak_intensities_range, num_parabolas_per_line, x, num_lines):
    lines = []
    for _ in range(num_lines):
        line = np.zeros_like(x)
        for _ in range(num_parabolas_per_line):
            peak = np.random.uniform(low=x[0]+5, high=x[-1]-5)
            intensity = np.random.uniform(*peak_intensities_range)
            a = -intensity / (10 ** 2)
            parabola = a * (x - peak) ** 2 + intensity
            parabola[parabola < 0] = 0
            line += parabola
        lines.append(line)
    return lines

# Generate 60 lines with multiple small inverted parabolas
small_parabolas_lines = generate_parabolas((0, 50), 5, x, 20)

# Generate 60 lines each with a large inverted parabola
large_parabolas_lines = generate_parabolas((100, 1000), 1, x, 20)

# Combine small parabolas lines and large parabolas lines into one dataset
data = np.concatenate((small_parabolas_lines, large_parabolas_lines))

# Create a DataFrame from the data_reference
df = pd.DataFrame(data, columns=x)

# Transpose the DataFrame to make X-axis values the columns and Y-axis values the rows
df = df.T

# Save the DataFrame to a CSV file, including the index
df.to_csv('blank_samples.csv', index_label='X-axis')