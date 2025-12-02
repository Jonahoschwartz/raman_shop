![alt text](../pngs/Nori.png "Title")

## Plate Reader Data Analysis and Visualization

**Tools for High-Throughput Screening Workflows**

This package provides utilities for parsing BioTek plate reader output files, extracting kinetic and static measurements, and visualizing 96-well plate data.

---

## File Parsing

### `parse_plate_reader_data(file_path, output_format="wide")`
Parse BioTek plate reader output files to extract both static endpoint data and kinetic time-course measurements.

**Parameters:**
- `file_path`: Path to BioTek output file (typically `.txt` or `.csv`)
- `output_format`: Output structure - `"wide"` (plate layout) or `"long"` (tidy format)

**Returns:**
Dictionary with keys `"static"` and `"kinetic"`, each containing DataFrames organized by read label.

**Wide Format:**
- **Static**: Dict of DataFrames with rows A-H, columns 1-12
- **Kinetic**: Dict of DataFrames with time as index, wells as columns, plus `Temperature` column

**Long Format:**
- **Static**: Tidy DataFrame with columns `Row`, `Column`, `Well`, `Value`, `Read`
- **Kinetic**: Tidy DataFrame with columns `Time`, `Well`, `Value`, `Read`

**Example:**
```python
# Parse in wide format (default)
data = parse_plate_reader_data("experiment_001.txt")

# Access static endpoint reads
od600_df = data['static']['Read 1:600']  # 8x12 DataFrame
print(od600_df.loc['A', '1'])  # Value in well A1

# Access kinetic time series
kinetic_df = data['kinetic']['Read 2:GFP/485,528']
print(kinetic_df.head())  # Time points as rows, wells as columns

# Parse in long format for analysis
data_long = parse_plate_reader_data("experiment_001.txt", output_format="long")
static_tidy = data_long['static']  # Easy for groupby operations
```

**Features:**
- Automatically detects and separates static vs. kinetic reads
- Handles overflow values (`OVRFLW`) as `NaN`
- Preserves temperature readings in kinetic data
- Supports multiple read types per file (OD, fluorescence, luminescence)

---

## Data Extraction & Transformation

### `extract_timepoint(df, timestamp, output_format="wide")`
Extract a full 96-well plate layout at a specific time from kinetic data.

**Parameters:**
- `df`: Kinetic DataFrame (wide or long format)
- `timestamp`: Time point to extract (e.g., `"0:15:00"`)
- `output_format`: Return as `"wide"` (plate layout) or `"long"` (tidy)

**Returns:**
DataFrame in specified format with values at the requested timepoint.

**Example:**
```python
# From wide format kinetic data
kinetic_df = data['kinetic']['Read 1:600']
t15_plate = extract_timepoint(kinetic_df, "0:15:00", output_format="wide")
print(t15_plate)  # 8x12 plate layout at t=15 min

# From long format
kinetic_long = data_long['kinetic']
t15_long = extract_timepoint(kinetic_long, "0:15:00", output_format="long")
print(t15_long[['Well', 'Value']])  # Tidy format for plotting
```

---

### `csv_to_plate_format(csv_path_or_df, input_format='tall', well_col='Well', value_col='Value')`
Convert well-based CSV data into a plate-format DataFrame for visualization or analysis.

**Parameters:**
- `csv_path_or_df`: File path (string) or pandas DataFrame
- `input_format`: `'tall'` (one row per well) or `'long'` (wells in first row, values in second)
- `well_col`: Name of column containing well IDs (for tall format)
- `value_col`: Name of column containing measurements (for tall format)

**Returns:**
DataFrame with rows A-H and columns 1-12 representing the plate layout.

**Example:**
```python
# From tall format CSV
plate_df = csv_to_plate_format("samples.csv", input_format='tall', 
                               well_col='Well', value_col='Concentration')

# From long format (e.g., exported from LIMS)
plate_df = csv_to_plate_format("template.csv", input_format='long')

# From existing DataFrame
df = pd.DataFrame({'Well': ['A1', 'A2', 'B1'], 'OD': [0.5, 0.7, 0.4]})
plate = csv_to_plate_format(df, input_format='tall', 
                            well_col='Well', value_col='OD')
```

---

### `split_wells_to_rowcol(wells)`
Parse well IDs into separate row and column components.

**Parameters:**
- `wells`: List, Series, or array of well identifiers (e.g., `['A1', 'B12', 'H7']`)

**Returns:**
DataFrame with columns `well`, `row`, and `column`.

**Example:**
```python
wells = ['A1', 'B3', 'H12', 'D7']
parsed = split_wells_to_rowcol(wells)
print(parsed)
#    well row  column
# 0   A1   A       1
# 1   B3   B       3
# 2  H12   H      12
# 3   D7   D       7

# Use for filtering or grouping
parsed[parsed['row'].isin(['A', 'H'])]  # Edge rows only
parsed[parsed['column'] <= 6]  # Left half of plate
```

---

### `add_rowcol_from_wells(df, well_col="well")`
Add `Row` and `Column` columns to a DataFrame containing well IDs.

**Parameters:**
- `df`: DataFrame with well identifiers
- `well_col`: Name of column containing well IDs (default: `"well"`)

**Returns:**
Copy of DataFrame with added `Row` and `Column` columns.

**Example:**
```python
# Add row/column for grouping
df = pd.DataFrame({
    'well': ['A1', 'A2', 'B1', 'B2'],
    'fluorescence': [1200, 1350, 980, 1100]
})

df_expanded = add_rowcol_from_wells(df, well_col='well')
print(df_expanded)
#   well  fluorescence Row  Column
# 0   A1          1200   A       1
# 1   A2          1350   A       2
# 2   B1           980   B       1
# 3   B2          1100   B       2

# Now easy to analyze by row
df_expanded.groupby('Row')['fluorescence'].mean()
```

---

## Visualization

### `plot_plate_heatmap(df, well_col="well", value_col="Value", ...)`
Generate a publication-quality heatmap of 96-well plate data.

**Parameters:**
- `df`: DataFrame with well IDs and values (long format)
- `well_col`: Column name containing well identifiers (default: `"well"`)
- `value_col`: Column name containing measurement values (default: `"Value"`)
- `title`: Plot title (optional)
- `cmap`: Matplotlib colormap (default: `"viridis"`)
- `annot`: Show values in cells (default: `True`)
- `title_fontsize`: Title font size (default: 20)
- `tick_fontsize`: Axis tick font size (default: 10)
- `figsize`: Figure dimensions (default: `(10, 6)`)
- `outputpath`: Path to save figure as SVG (optional)

**Example:**
```python
# Basic heatmap
df = pd.DataFrame({
    'well': ['A1', 'A2', 'B1', 'B2'],
    'OD600': [0.45, 0.52, 0.38, 0.61]
})

plot_plate_heatmap(df, well_col='well', value_col='OD600',
                   title='Cell Density at 24h')

# Customized for presentation
plot_plate_heatmap(df, well_col='well', value_col='OD600',
                   title='Growth Inhibition Screen',
                   cmap='RdYlGn_r',  # Reverse red-yellow-green
                   annot=True,
                   figsize=(12, 8),
                   title_fontsize=24,
                   outputpath='screen_results.svg')

# From parsed plate reader data
data_long = parse_plate_reader_data("assay.txt", output_format="long")
od_data = data_long['static'][data_long['static']['Read'].str.contains('600')]
plot_plate_heatmap(od_data, well_col='Well', value_col='Value',
                   title='OD600 Endpoint')
```

**Features:**
- Automatically arranges wells into standard 8×12 plate layout
- Handles missing wells gracefully (shown in light grey)
- Supports all Matplotlib colormaps
- Publication-ready SVG output

---

### `plot_kinetic_grid(df, title=None, outputpath=None, ...)`
Create an 8×12 grid of kinetic traces showing time-course data for all wells simultaneously.

**Parameters:**
- `df`: Kinetic DataFrame (wide format or long format with `Read`, `Time`, `Well`, `Value`)
- `title`: Overall plot title (optional)
- `outputpath`: Path to save figure as SVG (optional)
- `line_color`: Color for all traces (default: `'black'`)
- `line_width`: Line width for traces (default: 1.5)
- `title_fontsize`: Main title font size (default: 48)
- `well_fontsize`: Individual well label font size (default: 8)
- `tick_fontsize`: Axis tick font size (default: 6)
- `figsize`: Figure dimensions (default: `(48, 32)`)
- `label_rotation`: X-axis label rotation in degrees (default: 45)

**Example:**
```python
# Parse kinetic data
data = parse_plate_reader_data("growth_curve.txt")
kinetic_df = data['kinetic']['Read 1:600']

# Basic grid plot
plot_kinetic_grid(kinetic_df, title='Growth Curves - All Strains')

# Customized for poster presentation
plot_kinetic_grid(kinetic_df,
                 title='48h Growth Kinetics at 37°C',
                 line_color='darkblue',
                 line_width=2,
                 title_fontsize=64,
                 figsize=(60, 40),
                 outputpath='growth_curves_poster.svg')

# From long format
data_long = parse_plate_reader_data("assay.txt", output_format="long")
kinetic_long = data_long['kinetic']
plot_kinetic_grid(kinetic_long, 
                 title='GFP Expression Over Time',
                 line_color='green')
```

**Features:**
- Generates 96 subplots arranged in standard plate orientation (A1 top-left, H12 bottom-right)
- Shared axes for easy comparison across wells
- Handles missing wells by hiding empty subplots
- Ideal for quality control and visual inspection of entire plates

---
