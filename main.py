import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

# In-memory state:
# freezeId -> {
#     "request": original freeze request,
#     "response": generated freeze response
# }
FREEZES: Dict[str, Dict[str, Any]] = {}

SAFE_INTEGER_MAX = 9007199254740991


# ============================================================
# JSON helpers
# ============================================================

class DuplicateKeyError(Exception):
    pass


def duplicate_check_pairs(pairs):
    """
    Parse JSON objects while keeping the last value for duplicate keys.

    We do not reject the entire HTTP request because of duplicate
    JSON keys. Candidate/file validation is handled separately.
    """
    result = {}

    for key, value in pairs:
        result[key] = value

    return result


def reject_nonfinite(value):
    """
    Reject NaN / Infinity.
    """
    raise ValueError("Non-finite JSON number")


async def parse_json_request(request: Request) -> Any:
    """
    Parse request JSON manually.
    """
    raw = await request.body()

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=duplicate_check_pairs,
        parse_constant=reject_nonfinite,
    )


def canonical_json(value: Any) -> bytes:
    """
    Compact UTF-8 JSON.

    Equivalent for our purposes to:
        UTF8(JSON.stringify(value))
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def utf8_sort_key(value: str) -> bytes:
    return value.encode("utf-8")


def unique_strings(values: Any) -> bool:
    if not isinstance(values, list):
        return False

    if any(
        not isinstance(value, str) or value == ""
        for value in values
    ):
        return False

    return len(values) == len(set(values))


def is_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INTEGER_MAX
    )


def is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def is_floor(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def sorted_unique_codes(codes: List[str]) -> List[str]:
    """
    Deduplicate and sort codes using UTF-8 byte ordering.
    """
    return sorted(
        set(codes),
        key=lambda code: code.encode("utf-8"),
    )


def round12(value: float) -> float:
    return round(value, 12)


# ============================================================
# FREEZE
# ============================================================

def build_inventory(
    files: Any
) -> Tuple[bool, List[Dict[str, Any]], Optional[int], Optional[str]]:

    # Candidate-level invalid files.
    if not isinstance(files, dict) or len(files) == 0:
        return False, [], None, None

    inventory = []

    for filename, text in files.items():

        # Filename must be a non-empty string.
        if not isinstance(filename, str) or filename == "":
            return False, [], None, None

        # File content is data and must be a string.
        if not isinstance(text, str):
            return False, [], None, None

        raw = text.encode("utf-8")

        inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    # Sort by UTF-8 filename.
    inventory.sort(
        key=lambda item: item["name"].encode("utf-8")
    )

    total_bytes = sum(
        item["bytes"]
        for item in inventory
    )

    package_digest = hashlib.sha256(
        json.dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    return True, inventory, total_bytes, package_digest


def validate_freeze_top_level(data):

    if not isinstance(data, dict):
        return False

    if data.get("phase") != "freeze":
        return False

    freeze_id = data.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or not freeze_id
        or len(freeze_id) > 128
    ):
        return False

    calibration = data.get("calibrationDigest")
    tokenizer = data.get("tokenizerDigest")

    if not isinstance(calibration, str) or not calibration:
        return False

    if not isinstance(tokenizer, str) or not tokenizer:
        return False

    allowed = data.get("allowedUnsupportedReasons")

    if not isinstance(allowed, list):
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in allowed
    ):
        return False

    if len(allowed) != len(set(allowed)):
        return False

    candidates = data.get("candidates")

    if not isinstance(candidates, list):
        return False

    if len(candidates) == 0:
        return False

    names = []

    for candidate in candidates:

        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not isinstance(name, str) or name == "":
            return False

        names.append(name)

    if len(names) != len(set(names)):
        return False

    return True


def build_freeze_response(data: Dict[str, Any]) -> Dict[str, Any]:
    calibration_digest = data["calibrationDigest"]
    tokenizer_digest = data["tokenizerDigest"]

    allowed_reasons = set(
        data["allowedUnsupportedReasons"]
    )

    results = []

    for candidate in data["candidates"]:

        name = candidate["name"]

        # --------------------------------------------------
        # Files / inventory
        # --------------------------------------------------

        files_valid, inventory, total_bytes, package_digest = (
            build_inventory(candidate.get("files"))
        )

        # --------------------------------------------------
        # Candidate reason evaluation
        # --------------------------------------------------

        reason_codes = []

        unsupported_reason = candidate.get(
            "unsupportedReason"
        )

        # --------------------------------------------------
        # Unsupported candidate
        # --------------------------------------------------

        if unsupported_reason is not None:

            if (
                isinstance(unsupported_reason, str)
                and unsupported_reason != ""
                and unsupported_reason in allowed_reasons
            ):
                # An explicitly allowed unsupported reason
                # makes the candidate unsupported.
                status = "unsupported"

            else:
                # Unsupported reason exists but is not allowed.
                status = "invalid"

                reason_codes.append(
                    "UNALLOWED_UNSUPPORTED_REASON"
                )

                # Still evaluate the normal candidate constraints.
                if candidate.get("loadable") is not True:
                    reason_codes.append("NOT_LOADABLE")

                if candidate.get("calibrationDigest") != calibration_digest:
                    reason_codes.append("CALIBRATION_MISMATCH")

                if candidate.get("tokenizerDigest") != tokenizer_digest:
                    reason_codes.append("TOKENIZER_MISMATCH")

        # --------------------------------------------------
        # Normal candidate
        # --------------------------------------------------

        else:

            if candidate.get("loadable") is not True:
                reason_codes.append("NOT_LOADABLE")

            if candidate.get("calibrationDigest") != calibration_digest:
                reason_codes.append("CALIBRATION_MISMATCH")

            if candidate.get("tokenizerDigest") != tokenizer_digest:
                reason_codes.append("TOKENIZER_MISMATCH")

            if reason_codes:
                status = "invalid"
            else:
                status = "frozen"

        # --------------------------------------------------
        # Invalid files always invalidate the candidate.
        # --------------------------------------------------

        if not files_valid:
            status = "invalid"
            reason_codes.append("INVALID_INPUT")

            inventory = []
            total_bytes = None
            package_digest = None

        results.append({
            "name": name,
            "status": status,
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": sorted_unique_codes(reason_codes),
        })

    # ------------------------------------------------------
    # Sort candidates by UTF-8 candidate name
    # ------------------------------------------------------

    results.sort(
        key=lambda item: item["name"].encode("utf-8")
    )

    return {
        "freezeId": data["freezeId"],
        "candidates": results,
    }

# ============================================================
# SELECT
# ============================================================

def validate_policy(
    policy: Any,
    candidate_names: List[str],
) -> bool:

    if not isinstance(policy, dict):
        return False

    max_bytes = policy.get("maxBytes")
    aggregate_floor = policy.get(
        "aggregateFloor"
    )
    required_slices = policy.get(
        "requiredSlices"
    )
    max_latency = policy.get(
        "maxLatencyMs"
    )
    candidate_order = policy.get(
        "candidateOrder"
    )

    # maxBytes:
    # non-negative safe integer
    if not is_safe_integer(max_bytes):
        return False

    # aggregateFloor:
    # finite number in [0, 1]
    if not is_floor(aggregate_floor):
        return False

    # requiredSlices:
    # must be an object
    if not isinstance(required_slices, dict):
        return False

    # latency ceiling:
    # finite and non-negative
    if not is_finite_nonnegative_number(
        max_latency
    ):
        return False

    # candidateOrder:
    # array of unique non-empty names
    if not unique_strings(candidate_order):
        return False

    # Candidate set must match exactly.
    if set(candidate_order) != set(
        candidate_names
    ):
        return False

    # Validate required slice names/floors.
    for slice_name, floor in (
        required_slices.items()
    ):
        if (
            not isinstance(slice_name, str)
            or slice_name == ""
        ):
            return False

        if not is_floor(floor):
            return False

    return True


def recompute_manifest(
    candidate: Dict[str, Any],
) -> bool:
    """
    Verify inventory structure, sorting, byte totals,
    and package digest.
    """

    inventory = candidate.get(
        "inventory"
    )

    if not isinstance(inventory, list):
        return False

    previous_name = None
    seen_names = set()

    total = 0
    normalized_inventory = []

    for item in inventory:

        if not isinstance(item, dict):
            return False

        # Exact inventory shape.
        if set(item.keys()) != {
            "name",
            "bytes",
            "sha256",
        }:
            return False

        name = item.get("name")
        byte_count = item.get("bytes")
        digest = item.get("sha256")

        if (
            not isinstance(name, str)
            or name == ""
        ):
            return False

        if not is_safe_integer(byte_count):
            return False

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                char not in "0123456789abcdef"
                for char in digest
            )
        ):
            return False

        # Duplicate filenames are invalid.
        if name in seen_names:
            return False

        seen_names.add(name)

        # Verify UTF-8 sort order.
        if previous_name is not None:
            if (
                name.encode("utf-8")
                < previous_name.encode("utf-8")
            ):
                return False

        previous_name = name

        normalized_inventory.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )

        total += byte_count

    # Verify totalBytes.
    if candidate.get("totalBytes") != total:
        return False

    # Verify packageDigest.
    expected_digest = sha256_json(
        normalized_inventory
    )

    if (
        candidate.get("packageDigest")
        != expected_digest
    ):
        return False

    return True


def validate_binary(value: Any) -> bool:
    """
    A prediction/label must be exactly integer 0 or 1.
    bool is rejected.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    )


def calculate_metrics(
    candidate_name: str,
    rows: List[Dict[str, Any]],
    required_slices: Dict[str, float],
) -> Tuple[
    Optional[float],
    Dict[str, Optional[float]],
    bool,
]:
    """
    Calculate aggregate and required-slice accuracies.

    Returns:
        aggregate
        slices
        predictions_valid
    """

    if len(rows) == 0:
        return (
            None,
            {
                name: None
                for name in required_slices
            },
            False,
        )

    correct = 0
    total = len(rows)

    slice_correct: Dict[str, int] = {}
    slice_total: Dict[str, int] = {}

    # --------------------------------------------------------
    # Validate all rows first.
    # --------------------------------------------------------
    for row in rows:

        if not isinstance(row, dict):
            return (
                None,
                {
                    name: None
                    for name in required_slices
                },
                False,
            )

        label = row.get("label")
        slice_name = row.get("slice")
        predictions = row.get(
            "predictions"
        )

        if not validate_binary(label):
            return (
                None,
                {
                    name: None
                    for name in required_slices
                },
                False,
            )

        if (
            not isinstance(slice_name, str)
            or slice_name == ""
        ):
            return (
                None,
                {
                    name: None
                    for name in required_slices
                },
                False,
            )

        if not isinstance(
            predictions,
            dict,
        ):
            return (
                None,
                {
                    name: None
                    for name in required_slices
                },
                False,
            )

        prediction = predictions.get(
            candidate_name
        )

        if not validate_binary(
            prediction
        ):
            return (
                None,
                {
                    name: None
                    for name in required_slices
                },
                False,
            )

        # Aggregate.
        if prediction == label:
            correct += 1

        # Slice totals.
        slice_total[slice_name] = (
            slice_total.get(
                slice_name,
                0,
            )
            + 1
        )

        if prediction == label:
            slice_correct[slice_name] = (
                slice_correct.get(
                    slice_name,
                    0,
                )
                + 1
            )
        else:
            slice_correct.setdefault(
                slice_name,
                0,
            )

    aggregate = round12(
        correct / total
    )

    slices: Dict[
        str,
        Optional[float],
    ] = {}

    for slice_name in required_slices:

        if slice_name not in slice_total:
            slices[slice_name] = None
        else:
            slices[slice_name] = round12(
                slice_correct[slice_name]
                / slice_total[slice_name]
            )

    return (
        aggregate,
        slices,
        True,
    )


def make_empty_slice_result(
    required_slices: Any,
) -> Dict[str, None]:
    if not isinstance(
        required_slices,
        dict,
    ):
        return {}

    return {
        slice_name: None
        for slice_name in required_slices
        if isinstance(slice_name, str)
    }


def select_candidates(
    data: Dict[str, Any],
) -> Dict[str, Any]:

    freeze_id = data[
        "freezeId"
    ]

    submitted_candidates = data[
        "candidates"
    ]

    policy = data[
        "policy"
    ]

    rows = data[
        "rows"
    ]

    stored = FREEZES.get(
        freeze_id
    )

    # ========================================================
    # No freeze found
    # ========================================================
    if stored is None:

        results = []

        for candidate in submitted_candidates:

            name = (
                candidate.get("name")
                if isinstance(
                    candidate,
                    dict,
                )
                else ""
            )

            results.append(
                {
                    "name": name,
                    "aggregate": None,
                    "slices": make_empty_slice_result(
                        policy.get(
                            "requiredSlices",
                            {},
                        )
                    ),
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": [
                        "NOT_FROZEN"
                    ],
                }
            )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None,
        }

    stored_candidates = (
        stored["response"][
            "candidates"
        ]
    )

    # ========================================================
    # Exact lineage
    # ========================================================
    if submitted_candidates != stored_candidates:

        results = []

        for candidate in submitted_candidates:

            name = (
                candidate.get("name")
                if isinstance(
                    candidate,
                    dict,
                )
                else ""
            )

            results.append(
                {
                    "name": name,
                    "aggregate": None,
                    "slices": make_empty_slice_result(
                        policy.get(
                            "requiredSlices",
                            {},
                        )
                    ),
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": [
                        "INVALID_LINEAGE"
                    ],
                }
            )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None,
        }

    candidate_names = [
        candidate["name"]
        for candidate in stored_candidates
    ]

    # ========================================================
    # Policy validation
    # ========================================================
    if not validate_policy(
        policy,
        candidate_names,
    ):

        results = []

        for candidate in stored_candidates:

            results.append(
                {
                    "name": candidate["name"],
                    "aggregate": None,
                    "slices": {},
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": [
                        "INVALID_POLICY"
                    ],
                }
            )

        results.sort(
            key=lambda item:
            item["name"].encode("utf-8")
        )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None,
        }

    candidate_order = policy[
        "candidateOrder"
    ]

    order_index = {
        name: index
        for index, name in enumerate(
            candidate_order
        )
    }

    required_slices = policy[
        "requiredSlices"
    ]

    latencies = data.get(
        "latencies"
    )

    if not isinstance(
        latencies,
        dict,
    ):
        latencies = {}

    results = []
    admitted_candidates = []

    # ========================================================
    # Evaluate candidates
    # ========================================================
    for candidate in stored_candidates:

        name = candidate[
            "name"
        ]

        reason_codes = []

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------
        manifest_valid = recompute_manifest(
            candidate
        )

        if manifest_valid:
            total_bytes = candidate.get(
                "totalBytes"
            )
        else:
            total_bytes = None
            reason_codes.append(
                "INVALID_MANIFEST"
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------
        raw_latency = latencies.get(
            name
        )

        if is_finite_nonnegative_number(
            raw_latency
        ):
            latency_ms = raw_latency
        else:
            # Specification says return null when
            # latency cannot be validated.
            latency_ms = None

        # ----------------------------------------------------
        # Candidate must be frozen
        # ----------------------------------------------------
        if candidate.get(
            "status"
        ) != "frozen":
            reason_codes.append(
                "INVALID_LINEAGE"
            )

        # ----------------------------------------------------
        # Predictions / metrics
        # ----------------------------------------------------
        (
            aggregate,
            slices,
            predictions_valid,
        ) = calculate_metrics(
            name,
            rows,
            required_slices,
        )

        if not predictions_valid:
            reason_codes.append(
                "INVALID_PREDICTIONS"
            )

        # ----------------------------------------------------
        # Aggregate / slice floors
        # ----------------------------------------------------
        if predictions_valid:

            if (
                aggregate is None
                or aggregate
                < policy[
                    "aggregateFloor"
                ]
            ):
                reason_codes.append(
                    "AGGREGATE_FLOOR"
                )

            for (
                slice_name,
                floor,
            ) in required_slices.items():

                value = slices.get(
                    slice_name
                )

                if value is None:
                    reason_codes.append(
                        f"MISSING_SLICE:{slice_name}"
                    )

                elif value < floor:
                    reason_codes.append(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------
        if (
            total_bytes is not None
            and total_bytes
            > policy["maxBytes"]
        ):
            reason_codes.append(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # Latency limit
        # ----------------------------------------------------
        if (
            latency_ms is not None
            and latency_ms
            > policy["maxLatencyMs"]
        ):
            reason_codes.append(
                "LATENCY_LIMIT"
            )

        # ----------------------------------------------------
        # Final reason list
        # ----------------------------------------------------
        reason_codes = sorted_unique_codes(
            reason_codes
        )

        admitted = (
            len(reason_codes) == 0
        )

        result = {
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency_ms,
            "admitted": admitted,
            "reasonCodes": reason_codes,
        }

        results.append(result)

        if admitted:
            admitted_candidates.append(
                {
                    "name": name,
                    "totalBytes": total_bytes,
                    "latencyMs": latency_ms,
                    "order": order_index.get(
                        name,
                        len(order_index),
                    ),
                    "manifest": deepcopy(
                        candidate
                    ),
                }
            )

    # ========================================================
    # Result ordering
    #
    # First candidateOrder,
    # then UTF-8 name fallback.
    # ========================================================
    results.sort(
        key=lambda item: (
            order_index.get(
                item["name"],
                len(order_index),
            ),
            item["name"].encode(
                "utf-8"
            ),
        )
    )

    # ========================================================
    # Winner
    #
    # 1. smaller bytes
    # 2. lower latency
    # 3. candidate order
    # 4. UTF-8 name fallback
    # ========================================================
    if admitted_candidates:

        admitted_candidates.sort(
            key=lambda item: (
                item["totalBytes"],
                item["latencyMs"],
                item["order"],
                item["name"].encode(
                    "utf-8"
                ),
            )
        )

        winner = admitted_candidates[0]

    else:
        winner = None

    if winner is None:
        selected = None
        package_manifest = None
    else:
        selected = winner[
            "name"
        ]
        package_manifest = winner[
            "manifest"
        ]

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": package_manifest,
    }


# ============================================================
# /quantize
# ============================================================

@app.post("/quantize")
async def quantize(request: Request):

    # ========================================================
    # 1. Parse JSON
    # ========================================================
    try:
        data = await parse_json_request(request)

    except Exception as exc:
        print("JSON PARSE ERROR:", repr(exc))

        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # ========================================================
    # 2. Top-level request must be an object
    # ========================================================
    if not isinstance(data, dict):
        print("INVALID TOP LEVEL:", repr(data))

        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # ========================================================
    # 3. Phase must exist and be exactly freeze/select
    # ========================================================
    phase = data.get("phase")

    if phase not in ("freeze", "select"):
        print("INVALID PHASE:", repr(phase))

        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # ========================================================
    # 4. FREEZE
    # ========================================================
    if phase == "freeze":

        # ----------------------------------------------------
        # Validate GLOBAL freeze input.
        #
        # Candidate-level problems such as empty files,
        # unsupported reasons, bad loadability, etc. should
        # NOT cause HTTP 400. They belong in the candidate
        # result.
        # ----------------------------------------------------
        if not validate_freeze_top_level(data):

            print("FREEZE VALIDATION FAILED:")
            print(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        freeze_id = data["freezeId"]

        # ----------------------------------------------------
        # Existing freezeId
        # ----------------------------------------------------
        if freeze_id in FREEZES:

            stored = FREEZES[freeze_id]

            # Exact replay:
            # same freeze input -> return stored response unchanged.
            if stored["request"] == data:

                return JSONResponse(
                    status_code=200,
                    content=deepcopy(
                        stored["response"]
                    ),
                )

            # Same ID, different freeze input.
            return JSONResponse(
                status_code=409,
                content={
                    "error": "FREEZE_ID_CONFLICT"
                },
            )

        # ----------------------------------------------------
        # New valid freeze
        # ----------------------------------------------------
        response = build_freeze_response(data)

        # Store ONLY after global validation succeeds.
        # Therefore invalid freeze requests do not reserve IDs.
        FREEZES[freeze_id] = {
            "request": deepcopy(data),
            "response": deepcopy(response),
        }

        return JSONResponse(
            status_code=200,
            content=deepcopy(response),
        )

    # ========================================================
    # 5. SELECT
    # ========================================================

    # freezeId must be a non-empty string
    freeze_id = data.get("freezeId")

    # candidates must be an array
    candidates = data.get("candidates")

    # rows must be an array
    rows = data.get("rows")

    # policy must be an object
    policy = data.get("policy")

    if (
        not isinstance(freeze_id, str)
        or freeze_id == ""
        or not isinstance(candidates, list)
        or not isinstance(rows, list)
        or not isinstance(policy, dict)
    ):

        print("SELECT VALIDATION FAILED:")
        print(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )
        )

        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # --------------------------------------------------------
    # Selection request is structurally valid.
    # Let select_candidates() perform all candidate,
    # policy, manifest, prediction, and constraint checks.
    # --------------------------------------------------------
    response = select_candidates(data)

    return JSONResponse(
        status_code=200,
        content=deepcopy(response),
    )

# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service":
            "quantize-admit-api",
        "status": "ok",
    }
