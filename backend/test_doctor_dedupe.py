"""
Regression checks: doctor names must never repeat in recommendation payloads.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_loader import (  # noqa: E402
    LOCATION_COORDS,
    SPECIALTY_CANONICAL_ALIASES,
    _finalize_recommendation_doctor_lists,
    _normalize_doctor_name_key,
    build_closest_care_results,
    generate_recommendation_response,
    load_doctors_data,
    recommend_doctors,
)


def _assert_unique_doctor_names(doctors, context):
    names = [
        _normalize_doctor_name_key(doctor.get("name"))
        for doctor in doctors
        if isinstance(doctor, dict)
    ]
    names = [name for name in names if name]
    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    assert not duplicates, f"{context} has duplicate doctor names: {duplicates}"


def _collect_response_doctor_names(response):
    doctors = []
    doctors.extend(response.get("recommended_doctors") or [])
    optional = response.get("optional_nearby_doctors") or {}
    doctors.extend(optional.get("recommended_doctors") or [])
    for item in response.get("closest_results") or []:
        if str(item.get("type", "")).lower() == "doctor":
            doctors.append(item)
    return doctors


def test_csv_multi_specialty_names_collapse_per_query():
    df = load_doctors_data()
    multi_name = (
        df.assign(_name_key=df["name"].map(_normalize_doctor_name_key))
        .groupby("_name_key")
        .size()
    )
    assert (multi_name > 1).any(), "Fixture expectation: dataset should include multi-row names"

    for specialty in sorted(SPECIALTY_CANONICAL_ALIASES.keys()):
        for location in ("kothrud", "baner", "hadapsar", "wakad"):
            result = recommend_doctors(
                {"specialty": specialty, "location": location, "severity": "medium"}
            )
            _assert_unique_doctor_names(
                result.get("recommended_doctors") or [],
                f"recommend_doctors specialty={specialty} location={location}",
            )


def test_closest_and_full_response_have_unique_doctor_names():
    specialties = sorted(SPECIALTY_CANONICAL_ALIASES.keys())
    locations = sorted(LOCATION_COORDS.keys())[:12]

    for specialty in specialties:
        for location in locations:
            closest = build_closest_care_results(
                {"specialty": specialty, "location": location},
                top_k=10,
            )
            closest_doctors = [item for item in closest if item.get("type") == "doctor"]
            _assert_unique_doctor_names(
                closest_doctors,
                f"closest_results specialty={specialty} location={location}",
            )

            for severity in ("low", "medium", "high"):
                response = generate_recommendation_response(
                    {
                        "specialty": specialty,
                        "location": location,
                        "severity": severity,
                    }
                )
                if response.get("error"):
                    continue
                _assert_unique_doctor_names(
                    _collect_response_doctor_names(response),
                    f"full response specialty={specialty} location={location} severity={severity}",
                )


def test_finalize_strips_cross_list_duplicates():
    response = {
        "metadata": {"query_specialty": "general physician"},
        "recommended_doctors": [
            {
                "name": "Dr. Rama Joshirao Paranjape",
                "specialty": "general physician",
                "location": "karve nagar",
                "rating_score": 10,
                "consultation_fee": 500,
                "contact_number": "918037321133",
            }
        ],
        "optional_nearby_doctors": {
            "recommended_doctors": [
                {
                    "name": "Dr. Girish Date",
                    "specialty": "internal medicine",
                    "location": "deccan gymkhana",
                    "rating_score": 10,
                    "consultation_fee": 1100,
                    "contact_number": "912048552984",
                }
            ]
        },
        "closest_results": [
            {
                "type": "doctor",
                "name": "Dr. Rama Joshirao Paranjape",
                "specialty": "General Physician",
                "location": "Karve nagar",
                "contact_number": "918037321133",
            },
            {
                "type": "doctor",
                "name": "Dr. Girish Date",
                "specialty": "General Physician",
                "location": "deccan gymkhana",
                "contact_number": "912048552984",
            },
            {
                "type": "doctor",
                "name": "Dr. Anupama Joshi",
                "specialty": "General Physician",
                "location": "karve nagar",
                "contact_number": "911111111111",
            },
        ],
    }

    cleaned = _finalize_recommendation_doctor_lists(
        response,
        preferred_specialty="general physician",
    )
    names = [
        _normalize_doctor_name_key(doctor.get("name"))
        for doctor in _collect_response_doctor_names(cleaned)
    ]
    assert names.count("dr. rama joshirao paranjape") == 1
    assert names.count("dr. girish date") == 1
    assert "dr. anupama joshi" in names


if __name__ == "__main__":
    test_csv_multi_specialty_names_collapse_per_query()
    test_closest_and_full_response_have_unique_doctor_names()
    test_finalize_strips_cross_list_duplicates()
    print("All doctor dedupe checks passed.")
