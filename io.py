import csv
import numpy as np

NA_VALUES = {"", "nan", "na", "n/a", "null", "none", "?"}


def try_numeric(values, fill_value):
    result = np.empty(len(values), dtype=np.float64)
    for i, val in enumerate(values):
        s = val.strip().lower()
        if s in NA_VALUES:
            result[i] = fill_value
        else:
            try:
                result[i] = float(s)
            except ValueError:
                return None
    return result


def rows_to_dict(rows, header, fill_value, dtype):
    padded = [row + [""] * max(0, len(header) - len(row)) for row in rows]
    data = {}
    for j, col in enumerate(header):
        vals = [row[j] for row in padded]
        arr = try_numeric(vals, fill_value)
        if arr is None:
            arr = np.array(vals, dtype=object)
        elif dtype is not None:
            arr = arr.astype(dtype)
        data[col] = arr
    return data


def read_csv(filepath, delimiter=",", has_header=True, fill_value=np.nan, skip_rows=0, dtype=None, chunk_size=None):
    if not isinstance(filepath, str):
        raise TypeError(
            f"filepath has to be a string, got {type(filepath).__name__}")
    if chunk_size is not None:
        return read_chunked(filepath, delimiter, has_header, fill_value, skip_rows, dtype, chunk_size)
    try:
        fh = open(filepath, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find file: '{filepath}'")
    with fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                break
        header, rows = [], []
        for i, row in enumerate(reader):
            if i == 0 and has_header:
                header = [c.strip() for c in row]
            else:
                rows.append(row)
    if not rows:
        raise ValueError("no data rows found, is the file empty?")
    if not header:
        header = [f"col_{j}" for j in range(len(rows[0]))]
    return rows_to_dict(rows, header, fill_value, dtype)


def read_chunked(filepath, delimiter, has_header, fill_value, skip_rows, dtype, chunk_size):
    try:
        fh = open(filepath, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find file: '{filepath}'")
    with fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                return
        header, chunk, first = [], [], True
        for row in reader:
            if first and has_header:
                header = [c.strip() for c in row]
                first = False
                continue
            if first:
                header = [f"col_{j}" for j in range(len(row))]
                first = False
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield rows_to_dict(chunk, header, fill_value, dtype)
                chunk = []
        if chunk:
            yield rows_to_dict(chunk, header, fill_value, dtype)
