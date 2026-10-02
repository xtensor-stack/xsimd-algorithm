import io
import json

import polars as pl
import requests


def build_proxy_request(url: str) -> str:
    return "https://cors.io/?url=" + url


def download_file(url: str) -> dict[str, str | int | float]:
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def parse_response(raw: bytes) -> pl.DataFrame:
    data = json.loads(raw)["body"]
    return pl.DataFrame(json.loads(data)["benchmarks"])


def get_data_df(url: str) -> pl.DataFrame:
    url = build_proxy_request(url)
    data = download_file(url)
    return parse_response(data)


TIME_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def normalize_time_to_ns(df: pl.DataFrame) -> pl.DataFrame:
    factor = pl.col("time_unit").replace_strict(TIME_UNIT_TO_NS, return_dtype=pl.Float64)
    return df.with_columns(
        pl.col("real_time") * factor,
        pl.col("cpu_time") * factor,
        pl.lit("ns").alias("time_unit"),
    )


UNUSED_COLS = [
    "run_name",
    "family_index",
    "per_family_instance_index",
    "repetitions",
    "repetition_index",
    "threads",
    "aggregate_name",
    "aggregate_unit",
    "iterations",
    "bytes_per_second",
    "items_per_second",
]


def preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
            .drop(UNUSED_COLS)
            .filter(pl.col("run_type") == "iteration")
            .pipe(normalize_time_to_ns)
            .drop("run_type", "real_time", "time_unit")
    )


def cast_int_or_categorical(s: pl.Series) -> pl.Series:
    try:
        return s.cast(pl.Int32)
    except pl.exceptions.InvalidOperationError:
        return s.cast(pl.Categorical)


def create_df_categories(df: pl.DataFrame, pattern: str) -> pl.DataFrame:
    df = df.with_columns(pl.col("name").str.extract_groups(pattern).alias("name_parts"))
    parts = df["name_parts"].struct.fields
    df = df.unnest("name_parts").filter(~pl.all_horizontal(pl.col(parts).is_null()))
    return df.with_columns(cast_int_or_categorical(df[c]) for c in parts)


def get_preprocessed_df(url: str) -> pl.DataFrame:
    df = get_data_df(url)
    df = preprocess_df(df)
    return df
