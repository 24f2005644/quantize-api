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

SAFE_INTEGER_MAX = 9007199254740991  # JavaScript Number.MAX_SAFE_INTEGER


# ============================================================
# JSON helpers
# ============================================================

class DuplicateKeyError(Exception):
    pass


def duplicate_check_pairs(pairs):
    """
    Detect duplicate JSON object keys.

    Normal Python dict parsing silently keeps only the last duplicate
    key, which would make it impossible to enforce the assignment's
    "unique filenames" requirement reliably.
    """
    result = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"Duplicate key: {key}")
        result[key] = value

    return result


def reject_nonfinite(value):
    """
    Reject JSON constants such as NaN and Infinity.
    """
    raise ValueError("Non-finite JSON number")


async def parse_json_request(request: Request) -> Any:
    """
    Parse JSON manually so duplicate keys and malformed JSON
    can be rejected with the required 400 response.
    """
    try:
        raw = await request.body()

        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=duplicate_check_pairs,
            parse_constant=reject_nonfinite,
        )

    except (UnicodeDecodeError, json.JSONDecodeError,
            DuplicateKeyError, ValueError):
        raise


def canonical_json(value: Any) -> bytes:
    """
    Compact UTF-8 JSON similar to JSON.stringify(...).

    Important:
    - no spaces
    - UTF-8
    - preserve dictionary insertion order
    - do NOT ASCII-escape Unicode
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

    if any(not isinstance(x, str) or x == "" for x in values):
        return False

    return len(values) == len(set(values))


def is_safe_integer(value: Any) -> bool:
    """
    JavaScript-safe non-negative integer.
    """
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
    Deduplicate and sort by UTF-8 byte order.
    """
    unique = set(codes)
    return sorted(unique, key=lambda x: x.encode("utf-8"))


def round12(value: float) -> float:
    return round(value, 12)


# ============================================================
# Freeze helpers
# ============================================================

FREEZE_CODES = {
    "INVALID_INPUT",
    "UNALLOWED_UNSUPPORTED_REASON",
    "NOT_LOADABLE",
    "CALIBRATION_MISMATCH",
    "TOKENIZER_MISMATCH",
}


def build_inventory(files: Any) -> Tuple[bool, List[Dict[str, Any]], Optional[int], Optional[str]]:
    """
    Validate files and build:

    inventory = [
        {
            "name": "...",
            "bytes": ...,
            "sha256": "..."
        }
    ]

    sorted by UTF-8 filename.
    """

    if not isinstance(files, dict) or len(files) == 0:
        return False, [], None, None

    inventory = []

    for filename, text in files.items():

        # Filename must be a non-empty string.
        if not isinstance(filename, str) or filename == "":
            return False, [], None, None

        # File content must be a UTF-8 string.
        if not isinstance(text, str):
            return False, [], None, None

        encoded = text.encode("utf-8")

        inventory.append(
            {
                "name": filename,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )

    inventory.sort(key=lambda item: utf8_sort_key(item["name"]))

    total_bytes = sum(item["bytes"] for item in inventory)

    package_digest = sha256_json(inventory)

    return True, inventory, total_bytes, package_digest


def validate_freeze_top_level(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    if data.get("phase") != "freeze":
        return False

    freeze_id = data.get("freezeId")

    if (
        not isinstance(freeze_id, str)
        or freeze_id == ""
        or len(freeze_id) > 128
    ):
        return False

    calibration = data.get("calibrationDigest")
    tokenizer = data.get("tokenizerDigest")

    if not isinstance(calibration, str) or calibration == "":
        return False

    if not isinstance(tokenizer, str) or tokenizer == "":
        return False

    allowed = data.get("allowedUnsupportedReasons")

    if not unique_strings(allowed):
        return False

    candidates = data.get("candidates")

    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    return True


def build_freeze_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the deterministic freeze response.
    """

    calibration_digest = data["calibrationDigest"]
    tokenizer_digest = data["tokenizerDigest"]
    allowed_reasons = set(data["allowedUnsupportedReasons"])

    candidates = data["candidates"]

    # Candidate names must be unique and non-empty.
    names = []

    for candidate in candidates:
        if isinstance(candidate, dict):
            name = candidate.get("name")
        else:
            name = None

        if not isinstance(name, str) or name == "":
            continue

        names.append(name)

    duplicate_names = len(names) != len(set(names))

    results = []

    for candidate in candidates:

        # Default invalid candidate structure.
        if not isinstance(candidate, dict):
            results.append(
                {
                    "name": "",
                    "status": "invalid",
                    "inventory": [],
                    "totalBytes": None,
                    "packageDigest": None,
                    "reasonCodes": ["INVALID_INPUT"],
                }
            )
            continue

        name = candidate.get("name")

        # ----------------------------------------------------
        # Basic candidate validation
        # ----------------------------------------------------
        invalid_basic = (
            not isinstance(name, str)
            or name == ""
            or duplicate_names
        )

        files = candidate.get("files")

        if invalid_basic:
            results.append(
                {
                    "name": name if isinstance(name, str) else "",
                    "status": "invalid",
                    "inventory": [],
                    "totalBytes": None,
                    "packageDigest": None,
                    "reasonCodes": ["INVALID_INPUT"],
                }
            )
            continue

        files_valid, inventory, total_bytes, package_digest = build_inventory(files)

        if not files_valid:
            results.append(
                {
                    "name": name,
                    "status": "invalid",
                    "inventory": [],
                    "totalBytes": None,
                    "packageDigest": None,
                    "reasonCodes": ["INVALID_INPUT"],
                }
            )
            continue

        reason_codes = []

        unsupported_reason = candidate.get("unsupportedReason")

        # ----------------------------------------------------
        # Unsupported candidate
        # ----------------------------------------------------
        if unsupported_reason is not None:

            if (
                isinstance(unsupported_reason, str)
                and unsupported_reason != ""
                and unsupported_reason in allowed_reasons
            ):
                status = "unsupported"

            else:
                status = "invalid"
                reason_codes.append("UNALLOWED_UNSUPPORTED_REASON")

        # ----------------------------------------------------
        # Normal candidate
        # ----------------------------------------------------
        else:

            loadable = candidate.get("loadable")

            if loadable is not True:
                reason_codes.append("NOT_LOADABLE")

            if candidate.get("calibrationDigest") != calibration_digest:
                reason_codes.append("CALIBRATION_MISMATCH")

            if candidate.get("tokenizerDigest") != tokenizer_digest:
                reason_codes.append("TOKENIZER_MISMATCH")

            if reason_codes:
                status = "invalid"
            else:
                status = "frozen"

        results.append(
            {
                "name": name,
                "status": status,
                "inventory": inventory,
                "totalBytes": total_bytes,
                "packageDigest": package_digest,
                "reasonCodes": sorted_unique_codes(reason_codes),
            }
        )

    # Sort candidates by UTF-8 candidate name.
    results.sort(key=lambda x: x["name"].encode("utf-8"))

    return {
        "freezeId": data["freezeId"],
        "candidates": results,
    }


# ============================================================
# Select helpers
# ============================================================

SELECT_CODES = {
    "NOT_FROZEN",
    "INVALID_LINEAGE",
    "INVALID_POLICY",
    "INVALID_PREDICTIONS",
    "INVALID_MANIFEST",
    "AGGREGATE_FLOOR",
    "SIZE_LIMIT",
    "LATENCY_LIMIT",
}


def validate_policy(policy: Any, candidate_names: List[str]) -> bool:
    if not isinstance(policy, dict):
        return False

    max_bytes = policy.get("maxBytes")
    aggregate_floor = policy.get("aggregateFloor")
    required_slices = policy.get("requiredSlices")
    max_latency = policy.get("maxLatencyMs")
    candidate_order = policy.get("candidateOrder")

    if not is_safe_integer(max_bytes):
        return False

    if not is_floor(aggregate_floor):
        return False

    if not isinstance(required_slices, dict):
        return False

    if not is_finite_nonnegative_number(max_latency):
        return False

    if not isinstance(candidate_order, list):
        return False

    if not unique_strings(candidate_order):
        return False

    # Candidate order must be exactly the same set.
    if set(candidate_order) != set(candidate_names):
        return False

    # Required slice names must be non-empty strings.
    for slice_name, floor in required_slices.items():

        if not isinstance(slice_name, str) or slice_name == "":
            return False

        if not is_floor(floor):
            return False

    return True


def recompute_manifest(candidate: Dict[str, Any]) -> bool:
    """
    Verify:
      - inventory shape
      - byte totals
      - SHA-256 values
      - packageDigest
    """

    inventory = candidate.get("inventory")

    if not isinstance(inventory, list):
        return False

    recomputed_inventory = []

    previous_name = None

    for item in inventory:

        if not isinstance(item, dict):
            return False

        if set(item.keys()) != {"name", "bytes", "sha256"}:
            return False

        name = item.get("name")
        byte_count = item.get("bytes")
        digest = item.get("sha256")

        if not isinstance(name, str) or name == "":
            return False

        if not is_safe_integer(byte_count):
            return False

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            return False

        # Verify sorted order.
        if previous_name is not None:
            if name.encode("utf-8") < previous_name.encode("utf-8"):
                return False

        previous_name = name

        recomputed_inventory.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    # Inventory filenames must be unique.
    names = [x["name"] for x in recomputed_inventory]

    if len(names) != len(set(names)):
        return False

    # Recompute total.
    recomputed_total = sum(x["bytes"] for x in recomputed_inventory)

    if candidate.get("totalBytes") != recomputed_total:
        return False

    # Recompute package digest.
    recomputed_package_digest = sha256_json(recomputed_inventory)

    if candidate.get("packageDigest") != recomputed_package_digest:
        return False

    return True


def validate_binary(value: Any) -> bool:
    """
    Binary prediction/label:
    exactly 0 or 1, but bool is NOT accepted.
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
    bool
]:
    """
    Returns:
      aggregate
      slice accuracies
      predictions_valid
    """

    if not rows:
        return None, {name: None for name in required_slices}, False

    correct = 0
    total = len(rows)

    slice_correct: Dict[str, int] = {}
    slice_total: Dict[str, int] = {}

    # First validate all rows.
    for row in rows:

        if not isinstance(row, dict):
            return None, {name: None for name in required_slices}, False

        label = row.get("label")
        slice_name = row.get("slice")
        predictions = row.get("predictions")

        if not validate_binary(label):
            return None, {name: None for name in required_slices}, False

        if not isinstance(slice_name, str) or slice_name == "":
            return None, {name: None for name in required_slices}, False

        if not isinstance(predictions, dict):
            return None, {name: None for name in required_slices}, False

        prediction = predictions.get(candidate_name)

        if not validate_binary(prediction):
            return None, {name: None for name in required_slices}, False

        if prediction == label:
            correct += 1
            slice_correct[slice_name] = slice_correct.get(slice_name, 0) + 1
        else:
            slice_correct.setdefault(slice_name, 0)

        slice_total[slice_name] = slice_total.get(slice_name, 0) + 1

    aggregate = round12(correct / total)

    slices: Dict[str, Optional[float]] = {}

    for slice_name in required_slices:
        if slice_name not in slice_total:
            slices[slice_name] = None
        else:
            slices[slice_name] = round12(
                slice_correct[slice_name] / slice_total[slice_name]
            )

    return aggregate, slices, True


def result_code_sort_key(code: str):
    return code.encode("utf-8")


def select_candidates(data: Dict[str, Any]) -> Dict[str, Any]:
    freeze_id = data["freezeId"]
    submitted_candidates = data["candidates"]
    policy = data["policy"]
    rows = data["rows"]

    stored = FREEZES.get(freeze_id)

    # --------------------------------------------------------
    # Freeze does not exist
    # --------------------------------------------------------
    if stored is None:

        results = []

        for candidate in submitted_candidates:
            name = (
                candidate.get("name")
                if isinstance(candidate, dict)
                else ""
            )

            results.append(
                {
                    "name": name,
                    "aggregate": None,
                    "slices": {
                        slice_name: None
                        for slice_name in (
                            policy.get("requiredSlices", {})
                            if isinstance(policy, dict)
                            else {}
                        )
                    },
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": ["NOT_FROZEN"],
                }
            )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None,
        }

    stored_candidates = stored["response"]["candidates"]

    # --------------------------------------------------------
    # Exact lineage check
    # --------------------------------------------------------
    if submitted_candidates != stored_candidates:

        results = []

        for candidate in submitted_candidates:

            name = (
                candidate.get("name")
                if isinstance(candidate, dict)
                else ""
            )

            results.append(
                {
                    "name": name,
                    "aggregate": None,
                    "slices": {
                        slice_name: None
                        for slice_name in (
                            policy.get("requiredSlices", {})
                            if isinstance(policy, dict)
                            else {}
                        )
                    },
                    "totalBytes": None,
                    "latencyMs": None,
                    "admitted": False,
                    "reasonCodes": ["INVALID_LINEAGE"],
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

    policy_valid = validate_policy(policy, candidate_names)

    # Even if policy is invalid, produce deterministic results.
    if not policy_valid:

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
                    "reasonCodes": ["INVALID_POLICY"],
                }
            )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": sorted(
                results,
                key=lambda x: x["name"].encode("utf-8")
            ),
            "packageManifest": None,
        }

    candidate_order = policy["candidateOrder"]
    order_index = {
        name: index
        for index, name in enumerate(candidate_order)
    }

    required_slices = policy["requiredSlices"]

    latencies = data.get("latencies")

    if not isinstance(latencies, dict):
        latencies = {}

    results = []
    admitted_candidates = []

    for candidate in stored_candidates:

        name = candidate["name"]

        reason_codes = []

        # ----------------------------------------------------
        # Manifest validation
        # ----------------------------------------------------
        manifest_valid = recompute_manifest(candidate)

        total_bytes = (
            candidate.get("totalBytes")
            if manifest_valid
            else None
        )

        if not manifest_valid:
            reason_codes.append("INVALID_MANIFEST")

        # ----------------------------------------------------
        # Latency validation
        # ----------------------------------------------------
        raw_latency = latencies.get(name)

        if is_finite_nonnegative_number(raw_latency):
            latency_ms = raw_latency
        else:
            latency_ms = None
            reason_codes.append("INVALID_POLICY")

        # ----------------------------------------------------
        # Frozen status
        # ----------------------------------------------------
        if candidate.get("status") != "frozen":
            # Unsupported or invalid candidates cannot be admitted.
            if candidate.get("status") == "unsupported":
                reason_codes.append("INVALID_LINEAGE")
            else:
                reason_codes.append("INVALID_LINEAGE")

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------
        aggregate, slices, predictions_valid = calculate_metrics(
            name,
            rows,
            required_slices,
        )

        if not predictions_valid:
            reason_codes.append("INVALID_PREDICTIONS")

        # ----------------------------------------------------
        # Floors and slice checks
        # ----------------------------------------------------
        if predictions_valid:

            if aggregate is None or aggregate < policy["aggregateFloor"]:
                reason_codes.append("AGGREGATE_FLOOR")

            for slice_name, floor in required_slices.items():

                slice_accuracy = slices.get(slice_name)

                if slice_accuracy is None:
                    reason_codes.append(
                        f"MISSING_SLICE:{slice_name}"
                    )

                elif slice_accuracy < floor:
                    reason_codes.append(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------
        if (
            total_bytes is not None
            and total_bytes > policy["maxBytes"]
        ):
            reason_codes.append("SIZE_LIMIT")

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------
        if (
            latency_ms is not None
            and latency_ms > policy["maxLatencyMs"]
        ):
            reason_codes.append("LATENCY_LIMIT")

        reason_codes = sorted_unique_codes(reason_codes)

        admitted = len(reason_codes) == 0

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
                    "order": order_index.get(name, len(order_index)),
                    "manifest": deepcopy(candidate),
                }
            )

    # --------------------------------------------------------
    # Results order
    # --------------------------------------------------------
    results.sort(
        key=lambda item: (
            order_index.get(item["name"], len(order_index)),
            item["name"].encode("utf-8"),
        )
    )

    # --------------------------------------------------------
    # Winner
    # --------------------------------------------------------
    winner = None

    if admitted_candidates:

        admitted_candidates.sort(
            key=lambda item: (
                item["totalBytes"],
                item["latencyMs"],
                item["order"],
                item["name"].encode("utf-8"),
            )
        )

        winner = admitted_candidates[0]

    if winner is None:
        selected = None
        package_manifest = None
    else:
        selected = winner["name"]
        package_manifest = winner["manifest"]

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": package_manifest,
    }


# ============================================================
# Main endpoint
# ============================================================

@app.post("/quantize")
async def quantize(request: Request):

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------
    try:
        data = await parse_json_request(request)

    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # --------------------------------------------------------
    # Top-level input must be an object
    # --------------------------------------------------------
    if not isinstance(data, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    phase = data.get("phase")

    # --------------------------------------------------------
    # Unknown / missing phase
    # --------------------------------------------------------
    if phase not in ("freeze", "select"):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # ========================================================
    # FREEZE
    # ========================================================
    if phase == "freeze":

        if not validate_freeze_top_level(data):

            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_INPUT"},
            )

        freeze_id = data["freezeId"]

        # ----------------------------------------------------
        # Existing freeze ID
        # ----------------------------------------------------
        if freeze_id in FREEZES:

            stored = FREEZES[freeze_id]

            if stored["request"] == data:

                # Identical replay.
                return JSONResponse(
                    status_code=200,
                    content=deepcopy(stored["response"]),
                )

            # Same ID but different request.
            return JSONResponse(
                status_code=409,
                content={"error": "FREEZE_ID_CONFLICT"},
            )

        # ----------------------------------------------------
        # Create new frozen snapshot
        # ----------------------------------------------------
        response = build_freeze_response(data)

        FREEZES[freeze_id] = {
            "request": deepcopy(data),
            "response": deepcopy(response),
        }

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # ========================================================
    # SELECT
    # ========================================================

    # Required select structure.
    if (
        not isinstance(data.get("freezeId"), str)
        or data.get("freezeId") == ""
        or not isinstance(data.get("candidates"), list)
        or not isinstance(data.get("rows"), list)
        or not isinstance(data.get("policy"), dict)
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    response = select_candidates(data)

    return JSONResponse(
        status_code=200,
        content=response,
    )


# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "quantize-admit-api",
        "status": "ok",
    }
