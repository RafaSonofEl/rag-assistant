from urllib.parse import quote
from typing import Any, List, Union
from datetime import datetime
import re


def encode_uri_component(v: str) -> str:
    return quote(v, safe="~()*!'")


encoded_slash = encode_uri_component("/")
encoded_count = encode_uri_component("$count")
trailing_count_regex = re.compile(f"(?:(?:{encoded_slash})|/){encoded_count}$")


def escape_resource(resource: Union[str, List[str]]) -> str:
    if isinstance(resource, str):
        resource = encode_uri_component(
            resource,
        )
    else:
        resource = "/".join(map(encode_uri_component, resource))

    return trailing_count_regex.sub("/$count", resource)


def escape_parameter_alias(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"Parameter alias reference must be a string, got: {type(value)}"
        )

    return f"@{encode_uri_component(value)}"


def iso_format(dt: datetime):
    try:
        utc: Any = dt + dt.utcoffset()  # type: ignore
    except TypeError:
        utc: Any = dt

    isostring = datetime.strftime(utc, "%Y-%m-%dT%H:%M:%S.{0}Z")
    return isostring.format(int(round(utc.microsecond / 1000.0)))


def escape_value(value: Any) -> Union[str, int, float, bool, None]:
    if isinstance(value, str):
        value = value.replace("'", "''")
        return f"'{encode_uri_component(value)}'"
    elif isinstance(value, datetime):
        return f"datetime'{iso_format(value)}'"
    return value


# TODO: When we drop support for python 3.8/3.9 we can use
# Callable typing https://peps.python.org/pep-0677/ with a TypedVar("T") generic
def map_obj(obj: Any, fn: Any) -> Any:
    return [fn(value, key) for key, value in obj.items()]


def is_valid_option(key: str) -> bool:
    return key in [
        "$select",
        "$filter",
        "$expand",
        "$orderby",
        "$top",
        "$skip",
        "$format",
    ]


def bracket_join(arr: List[List[str]], separator: str) -> List[str]:
    if len(arr) == 1:
        return arr[0]

    result_arr: List[str] = []
    map_arr = map(
        lambda sub_arr: f"({''.join(sub_arr)})" if len(sub_arr) > 1 else sub_arr[0],
        arr,
    )
    for i, the_str in enumerate(map_arr):
        if i != 0:
            result_arr.append(separator)
        result_arr.append(the_str)

    return result_arr


def join(str_or_arr: Union[str, List[str]], separator: str = ",") -> str:
    if isinstance(str_or_arr, str):
        return str_or_arr
    elif isinstance(str_or_arr, list):  # type: ignore
        return separator.join(str_or_arr)
    else:
        raise Exception(f"Expected a string or array, got: {type(str_or_arr)}")
