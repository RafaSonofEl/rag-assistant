from typing import Any, Dict, Union, Literal, TypedDict, Optional, cast, List, Callable
from typing_extensions import NotRequired
from datetime import datetime
import re

from abc import ABC, abstractmethod

from .utils import (
    escape_resource,
    escape_parameter_alias,
    map_obj,
    escape_value,
    is_valid_option,
    bracket_join,
    join,
)

AnyObject = Dict[str, Any]

ODataMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

BaseResourceId = Union[str, int, float, datetime, Dict[Literal["@"], str]]
ResourceAlternateKey = Dict[str, BaseResourceId]
ResourceId = Union[BaseResourceId, ResourceAlternateKey]

FilterBaseType = Union[str, int, float, datetime, bool, None]

Primitive = Union[None, str, int, float, bool, datetime]

Filter = Union["FilterObj", "FilterArray", FilterBaseType]
FilterOperationValue = Filter
FilterFunctionValue = Filter
FilterArray = List[Filter]

# TODO: improve raw filter typing as it can be more than a string and use $string
RawFilter = str

Lambda = TypedDict("Lambda", {"$alias": str, "$expr": Filter})

DurationValue = TypedDict(
    "DurationValue",
    {
        "negative": bool,
        "days": float,
        "hours": float,
        "minutes": float,
        "seconds": float,
    },
    total=False,
)

FilterObj = TypedDict(
    "FilterObj",
    {
        "@": str,
        "$raw": RawFilter,
        "$": Union[str, List[str]],
        "$count": Filter,
        "$and": Filter,
        "$or": Filter,
        "$in": Filter,
        "$not": Filter,
        "$any": Lambda,
        "$all": Lambda,
        "$ne": FilterOperationValue,
        "$eq": FilterOperationValue,
        "$gt": FilterOperationValue,
        "$ge": FilterOperationValue,
        "$lt": FilterOperationValue,
        "$le": FilterOperationValue,
        "$add": FilterOperationValue,
        "$sub": FilterOperationValue,
        "$mul": FilterOperationValue,
        "$div": FilterOperationValue,
        "$mod": FilterOperationValue,
        "$contains": FilterFunctionValue,
        "$endswith": FilterFunctionValue,
        "$startswith": FilterFunctionValue,
        "$length": FilterFunctionValue,
        "$indexof": FilterFunctionValue,
        "$substring": FilterFunctionValue,
        "$tolower": FilterFunctionValue,
        "$toupper": FilterFunctionValue,
        "$trim": FilterFunctionValue,
        "$concat": FilterFunctionValue,
        "$year": FilterFunctionValue,
        "$month": FilterFunctionValue,
        "$day": FilterFunctionValue,
        "$hour": FilterFunctionValue,
        "$minute": FilterFunctionValue,
        "$second": FilterFunctionValue,
        "$fractionalseconds": FilterFunctionValue,
        "$date": FilterFunctionValue,
        "$time": FilterFunctionValue,
        "$totaloffsetminutes": FilterFunctionValue,
        "$now": FilterFunctionValue,
        "$duration": DurationValue,
        "$maxdatetime": FilterFunctionValue,
        "$mindatetime": FilterFunctionValue,
        "$totalseconds": FilterFunctionValue,
        "$round": FilterFunctionValue,
        "$floor": FilterFunctionValue,
        "$ceiling": FilterFunctionValue,
        "$isof": FilterFunctionValue,
        "$cast": FilterFunctionValue,
    },
    total=False,
)

FilterObjOperators = Literal[
    "@",
    "$raw",
    "$",
    "$count",
    "$and",
    "$or",
    "$in",
    "$not",
    "$any",
    "$all",
    "$ne",
    "$eq",
    "$gt",
    "$ge",
    "$lt",
    "$le",
    "$add",
    "$sub",
    "$mul",
    "$div",
    "$mod",
    "$contains",
    "$endswith",
    "$startswith",
    "$length",
    "$indexof",
    "$substring",
    "$tolower",
    "$toupper",
    "$trim",
    "$concat",
    "$year",
    "$month",
    "$day",
    "$hour",
    "$minute",
    "$second",
    "$fractionalseconds",
    "$date",
    "$time",
    "$totaloffsetminutes",
    "$now",
    "$maxdatetime",
    "$mindatetime",
    "$totalseconds",
    "$round",
    "$floor",
    "$ceiling",
    "$isof",
    "$cast",
]

FilterOperationKey = Literal[
    "$ne", "$eq", "$gt", "$ge", "$lt", "$le", "$add", "$sub", "$mul", "$div", "$mod"
]

FilterFunctionKey = Literal[
    "$contains",
    "$endswith",
    "$startswith",
    "$length",
    "$indexof",
    "$substring",
    "$tolower",
    "$toupper",
    "$trim",
    "$concat",
    "$year",
    "$month",
    "$day",
    "$hour",
    "$minute",
    "$second",
    "$fractionalseconds",
    "$date",
    "$time",
    "$totaloffsetminutes",
    "$now",
    "$maxdatetime",
    "$mindatetime",
    "$totalseconds",
    "$round",
    "$floor",
    "$ceiling",
    "$isof",
    "$cast",
]

ODataOptionCodeExampleMap = {
    "$filter": "$filter: a: $op: [b: $count: ... ]",
    "$expand": "$expand: a: $count: ...",
    "$orderby": "$orderby: { a: { $count: ... }, $dir: 'asc' }",
}

ResourceExpand = Dict[str, "ODataOptions"]
Expand = Union[str, ResourceExpand, List[Union[str, ResourceExpand]]]

OrderByDirection = Literal["asc", "desc"]

# TODO: advanced orderby with starting letters
OrderBy = Union[str, List["OrderBy"], Dict[str, OrderByDirection]]

ParameterAlias = Primitive

# ODataOptions = TypedDict("ODataOptions", {
#     "$filter": Filter,
#     "$count": "ODataOptions",
#     "$expand": Expand,
#     "$orderby": OrderBy,
#     "$top": int,
#     "$skip": int,
#     "$select": str | List[str],
#     "$format": str,
#     "$index": Index
# }, total=False)
ODataOptions = AnyObject

OptionTypes = Union[Filter, Expand, OrderBy, int, str, List[str], ParameterAlias]

duration_timepart_flag_entries = (
    ("hours", "H"),
    ("minutes", "M"),
    ("seconds", "S"),
)


class Params(TypedDict, total=False):
    api_prefix: str
    method: ODataMethod
    resource: str
    id: ResourceId
    url: str
    body: AnyObject
    passthrough: AnyObject
    passthrough_by_method: Dict[ODataMethod, AnyObject]
    options: ODataOptions


class GetOrCreateParams(TypedDict):
    api_prefix: NotRequired[str]
    resource: str
    id: ResourceAlternateKey
    url: NotRequired[str]
    body: AnyObject
    passthrough: NotRequired[AnyObject]
    passthrough_by_method: NotRequired[Dict[ODataMethod, AnyObject]]
    options: NotRequired[ODataOptions]


class UpsertParams(TypedDict):
    api_prefix: NotRequired[str]
    resource: str
    id: Dict[str, Primitive]
    url: NotRequired[str]
    body: AnyObject
    passthrough: NotRequired[AnyObject]
    passthrough_by_method: NotRequired[Dict[ODataMethod, AnyObject]]
    options: NotRequired[ODataOptions]


def is_primitive(obj: Any) -> bool:
    return obj is None or isinstance(obj, (str, int, float, bool, datetime))


def add_parent_key(
    filter: Union[List[str], str, bool, float, int, None],
    parent_key: Optional[List[str]] = None,
    operator: str = " eq ",
) -> List[str]:
    if parent_key is not None:
        if isinstance(filter, list):
            if len(filter) == 1:
                filter = filter[0]
            else:
                filter = f"({''.join(filter)})"
        else:
            if filter is None:
                filter_python_converted = "null"
            elif isinstance(filter, bool):
                filter_python_converted = "true" if filter else "false"
            else:
                filter_python_converted = filter

            filter = f"{filter_python_converted}"
        return [escape_resource(parent_key), operator, filter]
    if isinstance(filter, list):
        return filter
    return [f"{filter}"]


def handle_filter_array(
    filter: FilterArray, parent_key: Optional[List[str]] = None, min_elements: int = 2
) -> List[List[str]]:
    if len(filter) < min_elements:
        raise Exception(
            f"Filter arrays must have at least {min_elements}, got {filter}"
        )
    return list(map(lambda value: build_filter(value, parent_key), filter))


def filter_operation(
    filter: FilterOperationValue,
    operator: FilterOperationKey,
    parent_key: Optional[List[str]] = None,
) -> List[str]:
    op = " " + operator[1:] + " "
    if is_primitive(filter):
        filter_str = escape_value(filter)
        return add_parent_key(filter_str, parent_key, op)
    elif isinstance(filter, list):
        filter_arr = handle_filter_array(filter)
        filter_str = bracket_join(filter_arr, op)
        return add_parent_key(filter_str, parent_key)
    elif isinstance(filter, dict):
        result = handle_filter_object(filter)
        if len(result) < 1:
            raise Exception(
                f"{operator} objects must have at least 1 property, got: {filter}"
            )
        if len(result) == 1:
            return add_parent_key(result[0], parent_key, op)
        else:
            filter_str = bracket_join(result, op)
            return add_parent_key(filter_str, parent_key)
    else:
        raise Exception(f"Expected None/str/int/float/dict/list, got: {type(filter)}")


def filter_function(
    filter: FilterOperationValue,
    fn_identifier: FilterFunctionKey,
    parent_key: Optional[List[str]] = None,
) -> List[str]:
    fn_name = fn_identifier[1:]
    if is_primitive(filter):
        operands: List[Any] = []
        if parent_key is not None:
            operands.append(escape_resource(parent_key))
        operands.append(escape_value(filter))
        return [f"{fn_name}({','.join([o for o in operands if o is not None])})"]
    elif isinstance(filter, list):
        filter_arr = handle_filter_array(filter)
        filter_str = ",".join(map(lambda sub_arr: "".join(sub_arr), filter_arr))
        filter_str = f"{fn_name}({filter_str})"
        return add_parent_key(filter_str, parent_key)
    elif isinstance(filter, dict):
        filter_arr = handle_filter_object(filter)
        filter_str = ",".join(map(lambda sub_arr: "".join(sub_arr), filter_arr))
        filter_str = f"{fn_name}({filter_str})"
        return add_parent_key(filter_str, parent_key)
    else:
        raise Exception(f"Expected None/str/int/float/dict/list, got: {type(filter)}")


def apply_binds(
    filter: str,
    params: Dict[str, Filter],
    parent_key: Optional[List[str]] = None,
) -> List[str]:
    for index, param in params.items():
        param_str = f"({''.join(build_filter(param))})"
        # TODO: I am not yet sure about this regexes...
        param_str = param_str.replace("$", "$$")
        filter = re.sub(rf"\${index}([^a-zA-Z0-9]|$)", f"{param_str}\\1", filter)
    filter = f"({filter})"
    return add_parent_key(filter, parent_key)


def handle_expand_array(expands: List[Union[str, ResourceExpand]]) -> List[str]:
    if len(expands) < 1:
        raise Exception(f"Expand arrays must have at least 1 elements, got: {expands}")
    return list(map(lambda expand: build_expand(expand), expands))


def handle_expand_object(expand: ResourceExpand) -> List[str]:
    def __handle_expand_object(value: Any, key: str) -> str:
        if key[0] == "$":
            raise Exception(
                "Cannot have expand options without first expanding something!"
            )
        if is_primitive(value):
            raise Exception(
                f"'$expand: {key}: {value}' is invalid, use '$expand: {key}: $expand: ${value}' instead."
            )
        if isinstance(value, list):
            raise Exception(
                f"'$expand: {key}: [...]' is invalid, use '$expand: {key}: {{...}}' instead."
            )
        if key.endswith("/$count"):
            raise Exception(
                "'`$expand: { 'a/$count': {...} }` is deprecated, please use"
                "`$expand: { a: { $count: {...} } }` instead."
            )
        return handle_options("$expand", value, key)

    return map_obj(expand, __handle_expand_object)


def build_expand(expand: Expand) -> str:
    if is_primitive(expand):
        return escape_resource(expand)  # type: ignore
    elif isinstance(expand, list):
        expand_str = handle_expand_array(expand)
        return join(expand_str)
    elif isinstance(expand, dict):
        expand_str = handle_expand_object(expand)
        return join(expand_str)
    else:
        raise Exception(f"Unknown type for expand '${type(expand)}'")


def build_order_by(orderby: OrderBy) -> str:
    def __build_order_by(v: OrderBy) -> str:
        if isinstance(v, list):
            raise Exception("'$orderby' cannot have nested arrays")
        return build_order_by(v)

    if isinstance(orderby, str):
        if re.search(r"/\$count\b", orderby):
            raise Exception(
                "'`$orderby: 'a/$count'` is deprecated, please use `$orderby: { a: { $count: {...} } }`"
                "instead."
            )
        return orderby
    elif isinstance(orderby, list):
        if len(orderby) == 0:
            raise Exception("'$orderby' arrays have to have at least 1 element")
        result = map(__build_order_by, orderby)
        return join(list(result))
    elif isinstance(orderby, dict):  # type: ignore
        dollar_dir = orderby.get("$dir")

        remaining_keys = set(list(orderby.keys()))
        remaining_keys.discard("$dir")
        dollar_orderby = {k: orderby[k] for k in remaining_keys}

        def __map_dollar_orderby(dir_or_options: Any, key: str) -> str:
            property_path = key
            dir = dollar_dir
            if isinstance(dir_or_options, str):
                dir = dir_or_options
            else:
                keys = dir_or_options.keys()
                if "$count" not in dir_or_options or len(keys) > 1:
                    raise Exception(
                        f"When using {ODataOptionCodeExampleMap['$orderby']} you can only specify $count, "
                        f"got {list(keys)}"
                    )
                property_path = handle_options(
                    "$orderby", dir_or_options, property_path
                )
            if dir is None:  # type: ignore
                raise Exception(
                    "'$orderby' objects should either use the '{ a: 'asc' }' or the"
                    "'$orderby: { a: { $count: ... }, $dir: 'asc' }' notation"
                )
            if dir != "asc" and dir != "desc":
                raise Exception("'$orderby' direction must be 'asc' or 'desc'")
            return f"{property_path} {dir}"

        result = map_obj(dollar_orderby, __map_dollar_orderby)

        if len(result) != 1:
            raise Exception(
                f"'$orderby' objects must have exactly one element, got {len(result)} elements"
            )
        return result[0]
    else:
        raise Exception("'$orderby' option has to be either a string, array, or object")


def build_option(option: str, value: Any) -> str:
    compiled_value = ""
    if option == "$filter":
        compiled_value = "".join(build_filter(value))
    elif option == "$expand":
        compiled_value = build_expand(value)
    elif option == "$orderby":
        compiled_value = build_order_by(value)
    elif option in ["$top", "$skip"]:
        num = value
        if not isinstance(num, int):
            raise Exception(f"'{option}' option has to be a number")
        compiled_value += str(num)
    elif option == "$select":
        select = value
        if isinstance(select, str):
            compiled_value = join(select)
        elif isinstance(select, list):
            if len(select) == 0:  # type: ignore
                raise Exception(f"'{option}' arrays have to have at least 1 element")
            compiled_value = join(select)  # type: ignore
        else:
            raise Exception(f"'{option}' option has to be either a string or array")
    else:
        if option[0] == "@":
            if not is_primitive(value):
                raise Exception(
                    f"Unknown type for parameter alias option '{option}': {type(value)}"
                )
            compiled_value = str(escape_value(value))

        elif isinstance(value, list):
            compiled_value = join(value)  # type: ignore
        elif isinstance(value, str):
            compiled_value = value
        elif isinstance(value, bool):
            compiled_value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            compiled_value = str(value)

        else:
            raise Exception(f"Unknown type for option {type(value)}")

    return f"{option}={compiled_value}"


def handle_options(
    option_operation: Literal["$filter", "$expand", "$orderby"],
    options: ODataOptions,
    parent_key: str,
) -> str:
    if "$count" in options.keys():
        keys = options.keys()
        if len(keys) > 1:
            raise Exception(
                f"When using {ODataOptionCodeExampleMap[option_operation]}"
                f"you can only specify $count, got: {keys}"
            )
        options = options["$count"]
        parent_key += "/$count"

        count_filter = 1 if "$filter" in options.keys() else 0
        if len(options.keys()) > count_filter:
            if option_operation == "$expand":
                raise Exception(
                    "using OData options other than $filter in a '$expand: { a: { $count: {...} } }' "
                    "is not allowed, please remove them."
                )
            else:
                raise Exception(
                    f"When using {ODataOptionCodeExampleMap[option_operation]} "
                    f"you can only specify $filter in the $count, got {list(options.keys())}"
                )

    def __parse_options(value: Any, key: str) -> str:
        if key[0] == "$":
            if not is_valid_option(key):
                raise Exception(f"Unknown key option '{key}'")
            return build_option(key, value)
        if option_operation == "$expand":
            raise Exception(
                f"'$expand: {parent_key}: ${key}: ...' is invalid, use '$expand: {parent_key}: $expand:"
                f"{key}: ...' instead."
            )

        raise Exception(
            f"'${option_operation}: ${parent_key}: ${key}: ...' is invalid."
        )

    options_array = map_obj(options, __parse_options)
    options_str = ";".join(list(options_array))

    if len(options_str) > 0:
        options_str = f"({options_str})"

    return escape_resource(parent_key) + options_str


def handle_filter_operator(
    filter: FilterObj,
    operator: FilterObjOperators,
    parent_key: Optional[List[str]] = None,
) -> List[str]:
    if operator in [
        "$ne",
        "$eq",
        "$gt",
        "$ge",
        "$lt",
        "$le",
        "$add",
        "$sub",
        "$mul",
        "$div",
        "$mod",
    ]:
        return filter_operation(filter, cast(FilterOperationKey, operator), parent_key)
    elif operator in [
        "$contains",
        "$endswith",
        "$startswith",
        "$length",
        "$indexof",
        "$substring",
        "$tolower",
        "$toupper",
        "$trim",
        "$concat",
        "$year",
        "$month",
        "$day",
        "$hour",
        "$minute",
        "$second",
        "$fractionalseconds",
        "$date",
        "$time",
        "$totaloffsetminutes",
        "$now",
        "$maxdatetime",
        "$mindatetime",
        "$totalseconds",
        "$round",
        "$floor",
        "$ceiling",
        "$isof",
        "$cast",
    ]:
        return filter_function(filter, cast(FilterFunctionKey, operator), parent_key)

    elif operator == "$duration":
        if not isinstance(filter, dict):
            raise Exception(f"Expected type for {operator}, got: {type(filter)}")

        duration_value = cast(DurationValue, filter)
        duration_string = "P"

        days = duration_value.get("days")
        if days is not None:
            duration_string += f"{days}D"

        time_part = ""
        for part_key, part_flag in duration_timepart_flag_entries:
            key = duration_value.get(part_key)
            if key is not None:
                time_part += f"{key}{part_flag}"

        if len(time_part) > 0:
            duration_string += f"T{time_part}"

        if len(duration_string) <= 1:
            raise Exception(
                f"Expected {operator} to include duration properties, got: {type(filter)}"
            )

        if duration_value.get("negative"):
            duration_string = f"-{duration_string}"

        return add_parent_key(f"duration'{duration_string}'", parent_key)

    elif operator == "$raw":
        if isinstance(filter, str):
            f = f"({filter})"
            return add_parent_key(f, parent_key)
        elif not is_primitive(filter):
            if isinstance(filter, list):
                raw_filter = filter[0]  # type: ignore
                params = cast(List[Filter], filter[0:])  # type: ignore

                if not isinstance(raw_filter, str):
                    raise Exception(
                        f"First element of array for {operator} must be a string, got: {type(raw_filter)}"
                    )

                mapped_params: Dict[str, Filter] = {}
                for index in range(len(params)):
                    mapped_params[str(index)] = params[index]
                return apply_binds(raw_filter, mapped_params, parent_key)
            elif isinstance(filter, dict):  # type: ignore
                filter_str = filter["$string"]  # type: ignore
                if not isinstance(filter_str, str):
                    raise Exception(
                        f"$string element of object for {operator} must be a string got: "
                        f"{type(filter_str)}"  # type: ignore
                    )
                mapped_params: Dict[str, Filter] = {}
                for index in filter.keys():
                    if index != "$string":
                        if not re.match("^[a-zA-Z0-9]+$", index):
                            raise Exception(
                                f"{operator} param names must contain only [a-zA-Z0-9], got: {index}"
                            )
                        mapped_params[index] = cast(Filter, filter[index])
                return apply_binds(filter_str, mapped_params, parent_key)

            raise Exception(
                f"Expected None/str/int/float/dict/list, got: {type(filter)}"
            )

    elif operator == "$":
        resource = escape_resource(filter)  # type: ignore
        return add_parent_key(resource, parent_key)
    elif operator == "$count":
        keys = ["$count"]
        if (
            parent_key is not None
            and isinstance(filter, dict)  # type: ignore
            and (len(filter.keys()) == 0 or "$filter" in filter.keys())
        ):
            keys = parent_key[:-1]
            keys.append(handle_options("$filter", {"$count": filter}, parent_key[-1]))
            return ["/".join(keys)]
        if parent_key is not None:
            keys = parent_key + keys
        return build_filter(filter, keys)

    elif operator in ["$and", "$or"]:
        filter_str = build_filter(filter, None, f" {operator[1:]} ")
        return add_parent_key(filter_str, parent_key)
    elif operator == "$in":
        if is_primitive(filter):
            filter_str = escape_value(filter)
            return add_parent_key(filter_str, parent_key, " eq ")
        elif isinstance(filter, list):
            if all(map(is_primitive, filter)):
                filter_str = handle_filter_array(filter, None, 1)
                in_str = "".join(bracket_join(filter_str, ", "))
                return add_parent_key(f"({in_str})", parent_key, " in ")
            else:
                filter_str = handle_filter_array(filter, parent_key, 1)
                return bracket_join(filter_str, " or ")
        elif isinstance(filter, dict):  # type: ignore
            filter_arr = handle_filter_object(filter, parent_key)
            if len(filter_arr) < 1:
                raise Exception(
                    f"{operator} objects must have at least 1 property, got: {filter}"
                )
            return bracket_join(filter_arr, " or ")
        else:
            raise Exception(
                f"Expected None/str/int/float/dict/list, got: {type(filter)}"
            )
    elif operator == "$not":
        filter_str = f"not({''.join(build_filter(filter))})"
        return add_parent_key(filter_str, parent_key)
    elif operator in ["$any", "$all"]:
        alias = filter["$alias"]  # type: ignore
        expr = filter["$expr"]  # type: ignore

        if alias is None:
            raise Exception(f"Lambda expression ({operator}) has no alias defined.")
        if expr is None:
            raise Exception(f"Lambda expression ({operator}) has no expr defined.")

        filter_str = "".join(build_filter(expr))  # type: ignore
        filter_str = f"{operator[1:]}({alias}:{filter_str})"
        return add_parent_key(filter_str, parent_key, "/")

    raise Exception(f"Unrecognised operator: '{operator}'")


def handle_filter_object(
    filter: FilterObj,
    parent_key: Optional[List[str]] = None,
) -> List[List[str]]:
    def __handle_filter_object_key(
        value: Union[Filter, Lambda, None], key: str
    ) -> List[str]:
        # TODO: check None vs null+undefined here
        if key[0] == "$":
            return handle_filter_operator(value, key, parent_key)  # type: ignore
        elif key[0] == "@":
            parameter_alias = escape_parameter_alias(value)
            return add_parent_key(parameter_alias, parent_key)
        else:
            keys = [key]
            if parent_key is not None:
                # if len(parent_key) > 0:
                #     raise Exception(
                #         "`$filter: a: b: ...` is deprecated, please use `$filter: a: $any: { $alias: 'x', \
                #                     $expr: x: b: ... }` instead."
                #     )
                keys = parent_key + keys
            return build_filter(value, keys)  # type: ignore

    return map_obj(filter, __handle_filter_object_key)


def build_filter(
    filter: Filter,
    parent_key: Optional[List[str]] = None,
    join_str: Optional[str] = None,
) -> List[str]:
    if is_primitive(filter):
        filter_str = escape_value(filter)
        return add_parent_key(filter_str, parent_key)
    elif isinstance(filter, list):
        filter_arr = handle_filter_array(filter)
        filter_str = bracket_join(filter_arr, " or " if join_str is None else join_str)
        return add_parent_key(filter_str, parent_key)
    elif isinstance(filter, dict):
        filter_arr = handle_filter_object(filter, parent_key)
        return bracket_join(filter_arr, " and " if join_str is None else join_str)
    else:
        raise Exception(f"Expected None/str/int/float/dict/list, got: {type(filter)}")


class PinejsClientCore(ABC):
    def __init__(self, params: Union[str, Params]):
        params_to_set: Params = {}
        if isinstance(params, str):
            params_to_set = {"api_prefix": params}
        else:
            params_to_set = params

        self.api_prefix = params_to_set.get("api_prefix", "/")
        self.passthrough = params_to_set.get("passthrough", {})
        self.passthrough_by_method = params_to_set.get("passthrough_by_method", {})
        self.backend_params: Optional[AnyObject] = None

    def transform_get_result(self, params: Params) -> Callable[[Any], Any]:
        singular = False if params.get("id") is None else True

        def transform_get_result_fn(data: AnyObject) -> Any:
            if not isinstance(data, dict):  # type: ignore
                raise Exception(f"Response was not a JSON object: '{type(data)}'")
            data_d = data.get("d")
            if data_d is None:
                raise Exception(
                    "Invalid response received, the 'd' property is missing."
                )
            if singular:
                if len(data_d) > 1:
                    raise Exception(
                        "Returned multiple results when only one was expected."
                    )

                if len(data_d) == 0:
                    return None

                return data_d[0]
            return data_d

        return transform_get_result_fn

    def get(self, params: Params) -> Any:
        result = self.request({**params, "method": "GET"})
        return self.transform_get_result(params)(result)

    def put(self, params: Params) -> Any:
        return self.request({**params, "method": "PUT"})

    def patch(self, params: Params) -> Any:
        return self.request({**params, "method": "PATCH"})

    def post(self, params: Params) -> Any:
        return self.request({**params, "method": "POST"})

    def delete(self, params: Params) -> Any:
        return self.request({**params, "method": "DELETE"})

    def get_or_create(self, params: GetOrCreateParams) -> Any:
        id = params["id"]
        body = params["body"]

        if params["resource"].endswith("/$count"):
            raise Exception("getOrCreate does not support $count on resources")

        if body is None:  # type: ignore
            raise Exception("The body property is missing")

        if not isinstance(id, dict):  # type: ignore
            raise Exception(
                "The id property must be an object with the natural key of the model"
            )

        remaining_keys = set(list(params.keys()))
        remaining_keys.discard("id")
        remaining_keys.discard("body")
        remaining_params: Params = {k: params[k] for k in remaining_keys}  # type: ignore

        result = self.get({**remaining_params, "id": id})

        if result is not None:
            return result

        return self.post({**remaining_params, "body": {**id, **body}})

    def upsert(self, params: UpsertParams) -> Any:
        id = params["id"]
        body = params["body"]

        if not isinstance(id, dict):  # type: ignore
            raise Exception("The id property must be an object")

        natural_key_props = id.keys()
        if len(natural_key_props) == 0:
            raise Exception(
                "The id property must be an object with the natural key of the model"
            )

        if body is None:  # type: ignore
            raise Exception("The body property is missing")

        remaining_keys = set(list(params.keys()))
        remaining_keys.discard("id")
        remaining_keys.discard("body")
        remaining_params: Params = {k: params[k] for k in remaining_keys}  # type: ignore

        post_params: Params = {**remaining_params, "body": {**id, **body}}

        try:
            return self.post(post_params)
        except Exception as e:
            is_unique_violation_response = False
            if re.search(r"unique", e.message, re.IGNORECASE):  # type: ignore
                if e.status_code == 409:  # type: ignore
                    is_unique_violation_response = True

            if not is_unique_violation_response:
                raise e

            options = remaining_params.get("options", {})
            dollar_filter = (
                id
                if options.get("$filter") is None
                else {"$and": [options["$filter"], id]}
            )

            patch_parameters: Params = {
                **remaining_params,
                "options": {**options, "$filter": dollar_filter},
                "body": body,
            }

            return self.patch(patch_parameters)

    def request(self, params: Params) -> Any:
        # TODO: actually passthrought the passthrought stuff
        api_prefix = params.get("api_prefix", self.api_prefix)
        url = api_prefix + self.compile(params)
        method = params.get("method", "GET").upper()
        return self._request(method=method, url=url, body=params.get("body"))

    @abstractmethod
    def _request(self, method: str, url: str, body: Optional[Any] = None) -> Any:
        pass

    def compile(self, params: Params) -> str:
        url = params.get("url")

        if url is not None:
            return url

        resource = params.get("resource")
        if resource is None:
            raise Exception("Either the url or resource must be specified.")

        url = escape_resource(resource)
        options = params.get("options")

        if options is not None and options.get("$count") is not None:
            keys = options.keys()
            if len(keys) > 1:
                raise Exception(
                    f"When using '$expand: a: $count: ...' you can only specify $count, got: '{keys}'"
                )

            url += "/$count"
            options = options.get("$count")

        id = params.get("id")

        if id is not None:
            if isinstance(id, dict) and not isinstance(id, datetime):
                if "@" in id.keys():
                    value = escape_parameter_alias(id.get("@"))
                else:
                    value = ",".join(map_obj(id, self.__escape))
            else:
                v = escape_value(id)
                value = "" if v is None else str(v)

            url += f"({value})"

        query_options: List[str] = []

        if options is not None:
            query_options = map_obj(options, self.__validate_and_build_option)

        if len(query_options) > 0:
            url += "?" + "&".join(query_options)

        return url

    @staticmethod
    def __escape(v: BaseResourceId, k: str) -> str:
        if isinstance(v, dict) and "@" in v.keys():
            escaped_value = escape_parameter_alias(v.get("@"))
        else:
            escaped_value = str(escape_value(v))

        return f"{k}={escaped_value}"

    @staticmethod
    def __validate_and_build_option(value: OptionTypes, key: str) -> str:
        if key[0] == "$" and not is_valid_option(key):
            raise Exception(f"Unknown key option '{key}'")
        return build_option(key, value)
