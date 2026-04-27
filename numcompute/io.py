import csv
import numpy as np

NA_VALUES = {"", "nan", "na", "n/a", "null", "none", "?"}


def _try_numeric(values, fill_value):
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


def _rows_to_dict(rows, header, fill_value=np.nan, dtype=None):
    """
    Convert CSV rows into a dictionary of column arrays.
    """
    cleaned_rows = []

    for row in rows:
        if len(row) > len(header):
            raise ValueError("A row has more columns than the header.")

        padded_row = row + [""] * max(0, len(header) - len(row))
        cleaned_rows.append(padded_row)

    data = {}

    for j, col in enumerate(header):
        values = [row[j] for row in cleaned_rows]
        arr = _try_numeric(values, fill_value)

        if arr is None:
            arr = np.array(values, dtype=object)
        elif dtype is not None:
            try:
                arr = arr.astype(dtype)
            except ValueError as exc:
                raise ValueError(
                    f"Could not convert column '{col}' to dtype {dtype}."
                ) from exc

        data[col] = arr

    return data


def _dict_to_matrix(data):
    """
    Convert dictionary of column arrays into a 2D NumPy array.

    Only works when all columns are numeric.
    """
    columns = []

    for name, values in data.items():
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"Column '{name}' is non-numeric and cannot be converted to matrix."
            )

        columns.append(values)

    return np.column_stack(columns)


def read_csv(
    filepath,
    delimiter=",",
    has_header=True,
    fill_value=np.nan,
    skip_rows=0,
    dtype=None,
    chunk_size=None,
    return_dict=True,
):
    """
    Read a CSV file using plain Python and NumPy.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, default=','
        CSV delimiter.
    has_header : bool, default=True
        Whether the first non-skipped row contains column names.
    fill_value : scalar, default=np.nan
        Value used to replace missing entries.
    skip_rows : int, default=0
        Number of rows to skip before reading.
    dtype : data-type, optional
        Numeric dtype to convert numeric columns into.
    chunk_size : int, optional
        If provided, returns a generator that yields chunks.
    return_dict : bool, default=True
        If True, return dictionary of columns.
        If False, return 2D NumPy array. Only works for numeric data.

    Returns
    -------
    dict[str, np.ndarray] or np.ndarray
        CSV data as column dictionary or numeric matrix.

    Raises
    ------
    TypeError
        If filepath is not a string.
    ValueError
        If data is empty, chunk_size is invalid, or matrix conversion fails.
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}.")

    if chunk_size is not None:
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        return read_chunked(
            filepath=filepath,
            delimiter=delimiter,
            has_header=has_header,
            fill_value=fill_value,
            skip_rows=skip_rows,
            dtype=dtype,
            chunk_size=chunk_size,
            return_dict=return_dict,
        )

    try:
        file_handle = open(filepath, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: '{filepath}'")

    with file_handle:
        reader = csv.reader(file_handle, delimiter=delimiter)

        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                break

        header = []
        rows = []

        for i, row in enumerate(reader):
            if i == 0 and has_header:
                header = [col.strip() for col in row]
            else:
                rows.append(row)

    if not rows:
        raise ValueError("No data rows found. The file may be empty.")

    if not header:
        header = [f"col_{j}" for j in range(len(rows[0]))]

    data = _rows_to_dict(rows, header, fill_value, dtype)

    if return_dict:
        return data

    return _dict_to_matrix(data)


def read_chunked(
    filepath,
    delimiter=",",
    has_header=True,
    fill_value=np.nan,
    skip_rows=0,
    dtype=None,
    chunk_size=1000,
    return_dict=True,
):
    """
    Stream a CSV file in chunks.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    delimiter : str, default=','
        CSV delimiter.
    has_header : bool, default=True
        Whether the first non-skipped row contains column names.
    fill_value : scalar, default=np.nan
        Value used for missing entries.
    skip_rows : int, default=0
        Number of rows to skip.
    dtype : data-type, optional
        Numeric dtype for numeric columns.
    chunk_size : int, default=1000
        Number of rows per yielded chunk.
    return_dict : bool, default=True
        If True, yield dictionaries. If False, yield 2D numeric arrays.

    Yields
    ------
    dict[str, np.ndarray] or np.ndarray
        One chunk of CSV data.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    try:
        file_handle = open(filepath, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: '{filepath}'")

    with file_handle:
        reader = csv.reader(file_handle, delimiter=delimiter)

        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                return

        header = []
        chunk = []
        first = True

        for row in reader:
            if first and has_header:
                header = [col.strip() for col in row]
                first = False
                continue

            if first:
                header = [f"col_{j}" for j in range(len(row))]
                first = False

            chunk.append(row)

            if len(chunk) == chunk_size:
                data = _rows_to_dict(chunk, header, fill_value, dtype)
                yield data if return_dict else _dict_to_matrix(data)
                chunk = []

        if chunk:
            data = _rows_to_dict(chunk, header, fill_value, dtype)
            yield data if return_dict else _dict_to_matrix(data)