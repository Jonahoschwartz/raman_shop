import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict


# ============================================================
#                       PARSING FUNCTIONS
# ============================================================

def parse_plate_reader_data(file_path, output_format="wide"):
    """
    Parses BioTek plate reader output files to extract static and kinetic data.
    """
    import re
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    # Compile regex patterns
    TIME_PATTERN = re.compile(r"\d+:\d+:\d+")
    ROW_PATTERN = re.compile(r"^[A-H]\t")
    READ_LABEL_PATTERN = re.compile(r"(Read \d+:[\w/,]+)")

    def parse_value(v):
        """Convert string value to float, handling special cases."""
        if v is None:
            return np.nan
        v_stripped = v.strip() if isinstance(v, str) else str(v)
        if v_stripped in ("OVRFLW", ""):
            return np.nan
        try:
            return float(v_stripped)
        except (ValueError, AttributeError):
            return np.nan

    with open(file_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    # --- Find data section ---
    start = end = None
    for i, line in enumerate(lines):
        if "Results" in line:
            if start is None:
                start = i
            elif end is None:
                end = i
                break

    if start is None:
        raise ValueError("Could not find 'Results' section in file.")

    data_block = lines[start + 1:end] if end is not None else lines[start + 1:]

    def is_row_header(line):
        return bool(ROW_PATTERN.match(line.strip()))

    def is_time_header(line):
        return line.startswith("Time") and "Read" in line

    reads = defaultdict(lambda: defaultdict(list))
    kinetic_data = {}
    in_kinetic = False
    kinetic_label = None
    current_row_letter = None

    i = 0
    while i < len(data_block):
        line = data_block[i].strip()

        if not line:
            i += 1
            continue

        # --- Detect start of kinetic section ---
        if is_time_header(line):
            match = READ_LABEL_PATTERN.search(line)
            if not match:
                raise ValueError(f"Could not extract kinetic read label from: {line}")
            kinetic_label = match.group(1)
            col_headers = line.split('\t')[2:]
            kinetic_data[kinetic_label] = {"columns": col_headers, "data": []}
            in_kinetic = True
            i += 1
            continue

        # --- Kinetic data lines ---
        if in_kinetic and TIME_PATTERN.match(line):
            parts = line.split('\t')
            if len(parts) < 3:
                i += 1
                continue

            time = parts[0]
            temp = parse_value(parts[1])
            values = [parse_value(v) for v in parts[2:]]
            kinetic_data[kinetic_label]["data"].append((time, temp, values))
            i += 1
            continue

        # Exit kinetic mode
        if in_kinetic and not TIME_PATTERN.match(line):
            in_kinetic = False

        # --- Static section rows ---
        parts = line.split('\t')
        if len(parts) < 2:
            i += 1
            continue

        # Check if this line has a row letter (A–H)
        if is_row_header(line):
            current_row_letter = parts[0]
            row_values = [parse_value(v) for v in parts[1:-1]]
            read_label = parts[-1]
            row_values = (row_values + [np.nan] * 12)[:12]
            reads[read_label][current_row_letter] = row_values
        else:
            if READ_LABEL_PATTERN.match(parts[-1]) and current_row_letter:
                row_values = [parse_value(v) for v in parts[:-1]]
                read_label = parts[-1]
                row_values = (row_values + [np.nan] * 12)[:12]
                reads[read_label][current_row_letter] = row_values

        i += 1

    # --- Construct static DataFrames ---
    dfs_static = {}
    for read_label, rows_dict in reads.items():
        if not rows_dict:
            continue

        sorted_rows = sorted(rows_dict.items(), key=lambda x: x[0])
        row_letters = [r for r, _ in sorted_rows]
        data_matrix = [vals for _, vals in sorted_rows]

        df = pd.DataFrame(
            data=data_matrix,
            index=row_letters,
            columns=[str(i) for i in range(1, 13)]
        ).apply(pd.to_numeric, errors='coerce')

        dfs_static[read_label] = df

    # --- Construct kinetic DataFrames ---
    dfs_kinetic = {}
    for label, entry in kinetic_data.items():
        if not entry["data"]:
            continue

        times = [t for t, _, _ in entry["data"]]
        temps = [temp for _, temp, _ in entry["data"]]
        values = [v for _, _, v in entry["data"]]

        df = pd.DataFrame(
            data=values,
            columns=entry["columns"],
            index=times
        )
        df.index.name = "Time"
        df["Temperature"] = temps
        dfs_kinetic[label] = df

    # --- Output format ---
    if output_format == "wide":
        return {"static": dfs_static, "kinetic": dfs_kinetic}

    elif output_format == "long":
        static_long_list = []
        for label, df in dfs_static.items():
            df_reset = df.reset_index()
            df_melted = df_reset.melt(
                id_vars="index",
                var_name="Column",
                value_name="Value"
            )
            df_melted.rename(columns={"index": "Row"}, inplace=True)
            df_melted["Read"] = label
            df_melted["Well"] = df_melted["Row"] + df_melted["Column"]
            static_long_list.append(df_melted)

        static_long = pd.concat(static_long_list, ignore_index=True) if static_long_list else pd.DataFrame()

        kinetic_long_list = []
        for label, df in dfs_kinetic.items():
            df_temp = df.drop(columns="Temperature").reset_index()
            df_melted = df_temp.melt(
                id_vars="Time",
                var_name="Well",
                value_name="Value"
            )
            df_melted["Read"] = label
            kinetic_long_list.append(df_melted)

        kinetic_long = pd.concat(kinetic_long_list, ignore_index=True) if kinetic_long_list else pd.DataFrame()

        return {"static": static_long, "kinetic": kinetic_long}

    else:
        raise ValueError("output_format must be 'wide' or 'long'")


# ============================================================
#                EXTRACTION & TRANSFORM UTILITIES
# ============================================================

def extract_timepoint(df, timestamp, output_format="wide"):
    """
    Extracts a full plate layout at a specific timestamp.
    """
    if {"Read", "Time", "Well", "Value"}.issubset(df.columns):
        df_time = df[df["Time"] == timestamp]
        if df_time.empty:
            raise ValueError("Timestamp not found in 'Time' column.")

        df_time["Row"] = df_time["Well"].str[0]
        df_time["Column"] = df_time["Well"].str[1:]

        if output_format == "wide":
            plate_df = df_time.pivot(index="Row", columns="Column", values="Value")
            plate_df = plate_df.reindex(
                index=[chr(i) for i in range(65, 73)],
                columns=[str(i) for i in range(1, 13)]
            )
            return plate_df

        return df_time[["Row", "Column", "Value", "Well", "Read", "Time"]]

    else:
        if timestamp not in df.index:
            raise ValueError("Timestamp not found in DataFrame index.")

        row_data = df.loc[timestamp]
        well_data = {col: row_data[col] for col in df.columns if col != "Temperature"}

        plate_df = pd.DataFrame(
            index=[chr(i) for i in range(65, 73)],
            columns=[str(i) for i in range(1, 13)]
        )

        for well, value in well_data.items():
            row = well[0]
            col = well[1:]
            plate_df.at[row, col] = value

        if output_format == "wide":
            return plate_df

        plate_df = plate_df.stack().reset_index()
        plate_df.columns = ["Row", "Column", "Value"]
        plate_df["Well"] = plate_df["Row"] + plate_df["Column"]
        plate_df["Time"] = timestamp
        return plate_df[["Row", "Column", "Value", "Well", "Time"]]


def csv_to_plate_format(csv_path_or_df, input_format='tall', well_col='Well', value_col='Value'):
    """
    Convert tall or long CSV well data into a plate-format DataFrame.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df, header=None if input_format == 'wide' else 'infer')
    else:
        df = csv_path_or_df.copy()

    plate = pd.DataFrame(index=list('ABCDEFGH'), columns=range(1, 13))

    if input_format == 'tall':
        for _, row in df.iterrows():
            match = re.match(r'^([A-Ha-h])(\d{1,2})$', str(row[well_col]))
            if match:
                row_letter = match.group(1).upper()
                col_number = int(match.group(2))
                if row_letter in plate.index and col_number in plate.columns:
                    plate.at[row_letter, col_number] = row[value_col]

    elif input_format == 'long':
        wells = df.iloc[0].values
        values = df.iloc[1].values
        for well, value in zip(wells, values):
            match = re.match(r'^([A-Ha-h])(\d{1,2})$', str(well))
            if match:
                row_letter = match.group(1).upper()
                col_number = int(match.group(2))
                if row_letter in plate.index and col_number in plate.columns:
                    plate.at[row_letter, col_number] = value
    else:
        raise ValueError("input_format must be either 'tall' or 'wide'")

    return plate


def split_wells_to_rowcol(wells):
    """
    Takes a list or column of well names and returns row/column as a DataFrame.
    """
    rows = []
    cols = []
    for w in wells:
        match = re.match(r"^([A-Za-z]+)(\d+)$", str(w).strip())
        if match:
            rows.append(match.group(1).upper())
            cols.append(int(match.group(2)))
        else:
            rows.append(np.nan)
            cols.append(np.nan)
    return pd.DataFrame({"well": wells, "row": rows, "column": cols})


def add_rowcol_from_wells(df, well_col="well"):
    """
    Adds 'Row' and 'Column' columns from well IDs.
    """
    rows, cols = [], []

    for w in df[well_col]:
        match = re.match(r"^([A-Za-z]+)(\d+)$", str(w).strip())
        if match:
            rows.append(match.group(1).upper())
            cols.append(int(match.group(2)))
        else:
            rows.append(np.nan)
            cols.append(np.nan)

    df = df.copy()
    df["Row"] = rows
    df["Column"] = cols
    return df


# ============================================================
#                      PLOTTING FUNCTIONS
# ============================================================

def plot_plate_heatmap(df, well_col="well", value_col="Value", title=None,
                       cmap="viridis", annot=True, title_fontsize=20,
                       tick_fontsize=10, figsize=(10, 6), outputpath=None):
    """
    Plots a heatmap of a single plate.
    """
    rows, cols, values = [], [], []
    for w, v in zip(df[well_col], df[value_col]):
        match = re.match(r"^([A-Ha-h])(\d+)$", str(w).strip())
        if match:
            rows.append(match.group(1).upper())
            cols.append(match.group(2))
            values.append(v)
        else:
            rows.append(np.nan)
            cols.append(np.nan)
            values.append(np.nan)

    df_plate = pd.DataFrame({"Row": rows, "Column": cols, "Value": values})

    plate_df = df_plate.pivot_table(index="Row", columns="Column",
                                    values="Value", aggfunc="mean")

    plate_df = plate_df.reindex(index=[chr(i) for i in range(65, 73)],
                                columns=[str(i) for i in range(1, 13)])

    plt.figure(figsize=figsize)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")

    ax = sns.heatmap(plate_df, cmap=cmap_obj, annot=annot, fmt=".1f",
                     cbar=True, linewidths=0.5, linecolor="gray")

    if title:
        plt.title(title, fontsize=title_fontsize)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize, rotation=0)
    plt.xlabel("Column", fontsize=tick_fontsize)
    plt.ylabel("Row", fontsize=tick_fontsize)
    plt.tight_layout()

    if outputpath:
        plt.savefig(outputpath, bbox_inches="tight", format="svg")

    plt.show()


def plot_kinetic_grid(df, title=None, outputpath=None,
                      line_color='black', line_width=1.5,
                      title_fontsize=48, well_fontsize=8,
                      tick_fontsize=6, figsize=(48, 32),
                      label_rotation=45):
    """
    Plots an 8x12 grid of kinetic traces.
    """
    if {"Read", "Time", "Well", "Value"}.issubset(df.columns):
        df = df.pivot(index="Time", columns="Well", values="Value")

    fig, axes = plt.subplots(8, 12, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(title or "Kinetic Data by Well", fontsize=title_fontsize)

    for row in range(8):
        for col in range(12):
            ax = axes[row, col]
            well = chr(65 + row) + str(col + 1)

            if well in df.columns:
                ax.plot(df.index, df[well], color=line_color, linewidth=line_width)
                ax.set_title(well, fontsize=well_fontsize)
                ax.tick_params(labelsize=tick_fontsize)

                for label in ax.get_xticklabels():
                    label.set_rotation(label_rotation)
            else:
                ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if outputpath:
        plt.savefig(outputpath, bbox_inches="tight", format="svg")
    plt.show()
