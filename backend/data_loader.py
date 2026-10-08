import pandas as pd
from pathlib import Path
import json
import re

# Module-level cache to ensure file loads only once
_cached_data = None

# Default number of recommendations to return
DEFAULT_TOP_K = 10

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"


def _resolve_data_file(filename: str) -> Path:
    """
    Resolve dataset paths independent of process working directory.
    """
    candidates = [
        DATA_DIR / filename,
        Path("data") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


SPECIALTY_CANONICAL_ALIASES = {
    "general physician": {
        "general physician",
        "general medicine",
        "internal medicine",
        "general practitioner",
        "primary care",
        "primary care physician",
        "family physician",
        "gp",
        "physician",
    },
    "cardiologist": {
        "cardiologist",
        "cardiology",
        "cardiac specialist",
        "cardiac",
    },
    "dentists": {
        "dentist",
        "dentists",
        "dentistry",
        "dental",
        "dental surgeon",
    },
    "dermatologist": {
        "dermatologist",
        "dermatology",
        "skin specialist",
    },
    "endocrinologist": {
        "endocrinologist",
        "endocrinology",
    },
    "ent specialist": {
        "ent specialist",
        "ent",
        "ear nose throat",
        "otolaryngologist",
    },
    "gastroenterologist": {
        "gastroenterologist",
        "gastroenterology",
    },
    "gynac": {
        "gynac",
        "gynecologist",
        "gynaecologist",
        "gynecology",
        "obstetrics and gynaecology",
        "obgyn",
        "ob gyn",
        "obstetrician",
        "obsstetrician",
        "obstetrics",
    },
    "nephrologist": {
        "nephrologist",
        "nephrology",
    },
    "oncologist": {
        "oncologist",
        "oncology",
    },
    "ophthalmologist": {
        "ophthalmologist",
        "ophthalmology",
        "eye specialist",
        "eye",
    },
    "orthopedic": {
        "orthopedic",
        "orthopedics",
        "orthopaedic",
    },
    "pediatrician": {
        "pediatrician",
        "pediatrics",
        "paediatrician",
        "paediatrics",
    },
    "psychiatrist": {
        "psychiatrist",
        "psychiatry",
        "mental health specialist",
    },
    "pulmonologist": {
        "pulmonologist",
        "pulmonology",
        "respiratory",
        "chest physician",
    },
    "urologist": {
        "urologist",
        "urology",
    },
}

CANONICAL_SPECIALTY_DISPLAY = {
    "general physician": "General Physician",
    "cardiologist": "Cardiologist",
    "dentists": "Dentist",
    "dermatologist": "Dermatologist",
    "endocrinologist": "Endocrinologist",
    "ent specialist": "ENT Specialist",
    "gastroenterologist": "Gastroenterologist",
    "gynac": "Gynecologist",
    "nephrologist": "Nephrologist",
    "oncologist": "Oncologist",
    "ophthalmologist": "Ophthalmologist",
    "orthopedic": "Orthopedic",
    "pediatrician": "Pediatrician",
    "psychiatrist": "Psychiatrist",
    "pulmonologist": "Pulmonologist",
    "urologist": "Urologist",
}


def _normalize_specialty_text(value: str) -> str:
    """
    Normalize specialty text for robust matching.
    """
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_specialty(value: str) -> str:
    """
    Map specialty values from UI, doctor rows, and hospital rows
    to one canonical key.
    """
    normalized = _normalize_specialty_text(value)
    if not normalized:
        return ""

    for canonical, aliases in SPECIALTY_CANONICAL_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical

    for canonical, aliases in SPECIALTY_CANONICAL_ALIASES.items():
        for alias in aliases:
            if alias and alias in normalized:
                return canonical

    return normalized


def _display_specialty(canonical_specialty: str) -> str:
    """
    Convert canonical specialty key to UI display text.
    """
    canonical = canonicalize_specialty(canonical_specialty)
    if canonical in CANONICAL_SPECIALTY_DISPLAY:
        return CANONICAL_SPECIALTY_DISPLAY[canonical]
    return str(canonical_specialty or "General Physician").strip() or "General Physician"


def _normalize_doctor_name_key(value: str) -> str:
    """
    Normalize doctor display names for identity matching.
    """
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _doctor_contact_quality(contact_number) -> int:
    """
    Prefer rows that include a usable phone number.
    """
    text = str(contact_number or "").strip().lower()
    if not text or text in {"unavailable", "nan", "none", "not available"}:
        return 0
    digits = re.sub(r"\D", "", text)
    return 1 if len(digits) >= 8 else 0


def _dedupe_doctors_dataframe(df: pd.DataFrame, preferred_specialty: str = "") -> pd.DataFrame:
    """
    Keep one row per doctor name.

    Canonical specialty aliases (e.g. general physician + internal medicine, gynac +
    obstetrician) otherwise surface the same person twice in one result list.
    """
    if df is None or df.empty:
        return df

    if "name" not in df.columns:
        return df

    work = df.copy()
    preferred = canonicalize_specialty(preferred_specialty)
    work["_name_key"] = work["name"].map(_normalize_doctor_name_key)
    work = work[work["_name_key"].astype(str).str.len() > 0]
    if work.empty:
        return work.drop(columns=["_name_key"], errors="ignore")

    if "specialty" in work.columns:
        work["_specialty_canonical"] = work["specialty"].map(canonicalize_specialty)
        work["_specialty_raw"] = work["specialty"].astype(str).str.strip().str.lower()
        work["_preferred_specialty"] = work["_specialty_canonical"].eq(preferred).astype(int) if preferred else 0
        work["_exact_specialty"] = work["_specialty_raw"].eq(preferred).astype(int) if preferred else 0
    else:
        work["_preferred_specialty"] = 0
        work["_exact_specialty"] = 0

    if "rating_score" in work.columns:
        work["_rating_rank"] = pd.to_numeric(work["rating_score"], errors="coerce").fillna(0.0)
    else:
        work["_rating_rank"] = 0.0

    if "consultation_fee" in work.columns:
        work["_fee_rank"] = pd.to_numeric(work["consultation_fee"], errors="coerce").fillna(10**9)
    else:
        work["_fee_rank"] = 10**9

    if "contact_number" in work.columns:
        work["_contact_rank"] = work["contact_number"].map(_doctor_contact_quality)
    else:
        work["_contact_rank"] = 0

    work = work.sort_values(
        by=[
            "_name_key",
            "_preferred_specialty",
            "_exact_specialty",
            "_rating_rank",
            "_contact_rank",
            "_fee_rank",
        ],
        ascending=[True, False, False, False, False, True],
    )
    work = work.drop_duplicates(subset=["_name_key"], keep="first")
    return work.drop(
        columns=[
            "_name_key",
            "_specialty_canonical",
            "_specialty_raw",
            "_preferred_specialty",
            "_exact_specialty",
            "_rating_rank",
            "_fee_rank",
            "_contact_rank",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _dedupe_doctor_dicts(doctors: list, preferred_specialty: str = "") -> list:
    """
    Keep one dict per doctor name while preserving input preference order ties.
    """
    if not isinstance(doctors, list) or not doctors:
        return []

    preferred = canonicalize_specialty(preferred_specialty)
    best_by_name = {}
    order = []

    for index, doctor in enumerate(doctors):
        if not isinstance(doctor, dict):
            continue
        name_key = _normalize_doctor_name_key(doctor.get("name"))
        if not name_key:
            continue

        specialty_raw = str(doctor.get("specialty") or "").strip().lower()
        specialty_canonical = canonicalize_specialty(doctor.get("specialty"))
        rating = _to_float_or_none(doctor.get("rating_score"))
        if rating is None:
            rating = _to_float_or_none(doctor.get("rating")) or 0.0
        # rating_score may already be on 0-10 scale from rank_doctors
        if rating > 1.5:
            rating_rank = rating
        else:
            rating_rank = rating * 10
        contact_rank = _doctor_contact_quality(
            doctor.get("contact_number") or doctor.get("phone")
        )
        fee = _to_float_or_none(doctor.get("consultation_fee"))
        fee_rank = fee if fee is not None else 10**9
        preferred_rank = 1 if preferred and specialty_canonical == preferred else 0
        exact_rank = 1 if preferred and specialty_raw == preferred else 0
        score = (
            preferred_rank,
            exact_rank,
            rating_rank,
            contact_rank,
            -fee_rank,
            -index,
        )

        existing = best_by_name.get(name_key)
        if existing is None:
            best_by_name[name_key] = (score, doctor)
            order.append(name_key)
            continue

        if score > existing[0]:
            best_by_name[name_key] = (score, doctor)

    return [best_by_name[name_key][1] for name_key in order if name_key in best_by_name]


def _extract_hospital_specialty_keys(specialties_value: str) -> set:
    """
    Extract canonical specialty keys from a hospital specialties field.
    """
    values = str(specialties_value or "").replace("/", "|").replace(",", "|").split("|")
    result = set()
    for raw in values:
        canonical = canonicalize_specialty(raw)
        if canonical:
            result.add(canonical)
    return result


def _hospital_matches_specialty(specialties_value: str, requested_specialty: str) -> bool:
    """
    Check whether a hospital specialties string matches requested specialty.
    """
    requested = canonicalize_specialty(requested_specialty)
    if not requested:
        return False
    return requested in _extract_hospital_specialty_keys(specialties_value)

# Ordered nearby fallback map (lowercase keys/values)
NEARBY_MAP = {
    "akurdi": ["nigdi-pradhikaran", "pimpri-chinchwad", "ravet", "thergaon"],
    "alandi road": ["vishrantwadi", "dhanori", "yerwada", "lohegaon"],
    "amnora park town": ["hadapsar", "magarpatta city", "sasanenagar", "kharadi", "mundhwa", "keshav nagar"],
    "amanora park town": ["hadapsar", "magarpatta city", "sasanenagar", "kharadi", "mundhwa", "keshav nagar"],
    "anandnagar": ["sinhagad road", "vadgaon budruk", "manik bag", "hingne"],
    "aundh": ["baner", "pashan", "wakad", "balewadi", "pimple nilakh"],
    "balewadi": ["baner", "wakad", "aundh", "mahalunge", "pimple nilakh"],
    "baner": ["baner road", "aundh", "balewadi", "pashan", "wakad", "mahalunge"],
    "baner pashan link road": ["baner", "pashan", "balewadi"],
    "baner road": ["baner", "aundh", "balewadi", "pashan", "wakad"],
    "bavdhan": ["kothrud", "warje", "paud road", "pashan"],
    "bhosari": ["moshi", "pimpri-chinchwad", "chikhali", "dhanori"],
    "bibvewadi": ["market yard", "dhankawadi", "parvati gaon", "kondhwa", "swargate", "mukund nagar"],
    "bibwewadi kondhwa road": ["bibvewadi", "kondhwa", "market yard"],
    "budhwar peth": ["shaniwar peth", "kasba peth", "sadashiv peth", "camp"],
    "bund garden": ["koregaon park", "camp", "yerwada", "kalyani nagar", "dhole patil road", "viman nagar"],
    "camp": ["wanowrie", "wanwadi", "koregaon park", "kasba peth", "dhole patil road", "rasta peth"],
    "chakan": ["moshi", "bhosari", "chikhali", "akurdi", "nigdi-pradhikaran"],
    "chikhali": ["moshi", "bhosari", "talawade", "akurdi", "nigdi-pradhikaran"],
    "dhankawadi": ["katraj", "bibvewadi", "parvati gaon"],
    "dhanori": ["vishrantwadi", "lohegaon", "yerwada", "bhosari", "viman nagar", "kharadi"],
    "deccan gymkhana": ["fc road", "jm road", "erandwane", "shivajinagar", "prabhat road", "sadashiv peth"],
    "dhole patil road": ["camp", "koregaon park", "bund garden", "kasba peth", "rasta peth", "shivajinagar", "kalyani nagar", "viman nagar", "wadgaon sheri"],
    "erandwane": ["kothrud", "deccan gymkhana", "karve nagar", "law college road", "prabhat road", "senapati bapat marg"],
    "fc road": ["jm road", "deccan gymkhana", "shivajinagar", "model colony", "senapati bapat marg"],
    "hadapsar": ["magarpatta city", "kharadi", "sasanenagar", "wanowrie"],
    "hinjewadi": ["wakad", "mahalunge", "marunji"],
    "jm road": ["fc road", "shivajinagar", "deccan gymkhana", "model colony", "senapati bapat marg"],
    "junnar": ["chakan", "moshi", "bhosari"],
    "kalyani nagar": ["viman nagar", "yerwada", "wadgaon sheri", "koregaon park"],
    "karve nagar": ["kothrud", "warje", "erandwane"],
    "kasba peth": ["budhwar peth", "shaniwar peth", "camp", "rasta peth", "mangalvar peth", "sadashiv peth"],
    "katraj": ["dhankawadi", "ambegaon", "parvati gaon", "bibvewadi", "swargate", "market yard"],
    "keshav nagar": ["mundhwa", "kharadi", "magarpatta city", "hadapsar", "wadgaon sheri"],
    "kharadi": ["viman nagar", "hadapsar", "wagholi", "magarpatta city", "keshav nagar"],
    "kondhwa": ["bibvewadi", "nibm", "mohamadwadi", "lulla nagar"],
    "koregaon park": ["kalyani nagar", "yerwada", "camp", "bund garden", "dhole patil road", "viman nagar"],
    "kothrud": ["karve nagar", "paud road", "warje", "shivajinagar", "bavdhan"],
    "law college road": ["erandwane", "prabhat road", "deccan gymkhana", "kothrud", "model colony"],
    "lohegaon": ["viman nagar", "dhanori", "wagholi"],
    "lonavala": ["hinjewadi", "wakad", "pimpri-chinchwad", "ravet"],
    "lulla nagar": ["wanowrie", "kondhwa", "salunke vihar", "nibm", "mohamadwadi", "wanwadi"],
    "magarpatta city": ["hadapsar", "kharadi", "amnora park town", "mundhwa", "keshav nagar"],
    "mahalunge": ["balewadi", "baner", "hinjewadi"],
    "market yard": ["bibvewadi", "swargate", "mukund nagar"],
    "model colony": ["shivajinagar", "senapati bapat marg", "jm road", "fc road", "deccan gymkhana", "law college road"],
    "mohamadwadi": ["kondhwa", "undri", "nibm", "wanowrie", "hadapsar"],
    "moshi": ["bhosari", "chikhali", "pimpri-chinchwad"],
    "mukund nagar": ["swargate", "market yard", "parvati gaon"],
    "nagar road": ["kalyani nagar", "viman nagar", "yerwada"],
    "nanded city": ["sinhagad road", "vadgaon budruk"],
    "navi peth": ["sadashiv peth", "shaniwar peth"],
    "nibm": ["kondhwa", "undri", "wanowrie"],
    "nigdi-pradhikaran": ["akurdi", "ravet", "pimpri-chinchwad"],
    "parvati gaon": ["dhankawadi", "mukund nagar", "bibvewadi"],
    "pashan": ["baner", "aundh", "bavdhan", "balewadi", "wakad", "baner road"],
    "paud road": ["kothrud", "warje", "bavdhan"],
    "pimple nilakh": ["wakad", "aundh", "balewadi", "pimple saudagar"],
    "pimple saudagar": ["wakad", "pimple nilakh", "thergaon"],
    "pimpri-chinchwad": ["akurdi", "bhosari", "thergaon", "ravet"],
    "porwal road": ["lohegaon", "dhanori"],
    "prabhat road": ["law college road", "deccan gymkhana", "kothrud", "shivajinagar"],
    "ravet": ["akurdi", "nigdi-pradhikaran", "thergaon", "wakad", "pimpri-chinchwad"],
    "sadashiv peth": ["budhwar peth", "shaniwar peth", "navi peth", "narayan peth", "tilak road", "swargate"],
    "salunke vihar": ["wanowrie", "lulla nagar", "kondhwa", "nibm", "wanwadi"],
    "sasanenagar": ["hadapsar", "magarpatta city"],
    "senapati bapat marg": ["model colony", "shivajinagar", "fc road", "jm road", "deccan gymkhana"],
    "shaniwar peth": ["budhwar peth", "sadashiv peth", "kasba peth", "narayan peth", "tilak road", "rasta peth"],
    "shankar shet road": ["swargate", "mukund nagar", "market yard", "parvati gaon", "bibvewadi"],
    "shivajinagar": ["deccan gymkhana", "model colony", "fc road", "kothrud", "senapati bapat marg", "law college road", "prabhat road"],
    "sinhagad road": ["vadgaon budruk", "anandnagar", "nanded city", "dhayari", "karve nagar", "warje"],
    "sopan baug": ["wanowrie", "koregaon park", "camp", "wanwadi", "magarpatta city"],
    "swargate": ["market yard", "mukund nagar", "shankar shet road", "bibvewadi", "parvati gaon"],
    "thergaon": ["wakad", "ravet", "pimpri-chinchwad", "pimple saudagar", "bhosari"],
    "undri": ["kondhwa", "mohamadwadi", "nibm", "wanowrie", "mohammadwadi", "hadapsar"],
    "vadgaon budruk": ["sinhagad road", "nanded city", "dhayari", "anandnagar", "karve nagar"],
    "viman nagar": ["kalyani nagar", "kharadi", "lohegaon", "yerwada"],
    "vishrantwadi": ["dhanori", "yerwada", "alandi road"],
    "wadgaon sheri": ["kalyani nagar", "viman nagar", "kharadi"],
    "wagholi": ["kharadi", "lohegaon", "viman nagar", "wadgaon sheri"],
    "wakad": ["hinjewadi", "baner", "balewadi", "pimpri-chinchwad", "pimple nilakh"],
    "wanowrie": ["wanwadi", "camp", "lulla nagar", "sopan baug", "nibm", "mohamadwadi", "salunke vihar"],
    "wanwadi": ["wanowrie", "camp", "salunke vihar", "lulla nagar", "kondhwa", "sopan baug", "nibm"],
    "warje": ["kothrud", "karve nagar", "paud road", "bavdhan", "sinhagad road", "dhayari"],
    "yerwada": ["kalyani nagar", "viman nagar", "dhanori", "bund garden", "nagar road", "wadgaon sheri"],
    # Previously unmapped locations
    "bhugaon": ["bavdhan", "warje", "kothrud"],
    "bt kawade road": ["kharadi", "viman nagar", "wadgaon sheri"],
    "chandan nagar": ["kharadi", "viman nagar", "wadgaon sheri"],
    "dapodi": ["khadki", "aundh", "pimpri-chinchwad"],
    "dhayari": ["sinhagad road", "warje", "kothrud"],
    "ganga dham": ["market yard", "bibvewadi", "swargate"],
    "gultekdi": ["market yard", "swargate", "bibvewadi"],
    "khadki": ["dapodi", "aundh", "shivajinagar"],
    "magarpatta": ["hadapsar", "magarpatta city", "kharadi"],
    "manik bag": ["kothrud", "karve nagar", "erandwane"],
    "mangalvar peth": ["rasta peth", "kasba peth", "budhwar peth", "camp", "dhole patil road"],
    "marunji": ["hinjewadi", "mahalunge", "wakad"],
    "mohammadwadi": ["kondhwa", "undri", "nibm", "wanowrie", "hadapsar"],
    "mundhwa": ["keshav nagar", "kharadi", "hadapsar"],
    "narayan peth": ["sadashiv peth", "shaniwar peth", "budhwar peth"],
    "parvati paytha": ["parvati gaon", "swargate", "mukund nagar"],
    "rahatani": ["pimple saudagar", "thergaon", "wakad"],
    "rasta peth": ["kasba peth", "budhwar peth", "camp", "mangalvar peth"],
    "shukrawar peth": ["shaniwar peth", "budhwar peth", "sadashiv peth"],
    "tilak road": ["sadashiv peth", "shaniwar peth", "deccan gymkhana", "narayan peth", "swargate", "rasta peth"],
}


# Location coordinates map (lowercase keys) - Real coordinates from OpenStreetMap Nominatim API
LOCATION_COORDS = {
    "akurdi": (18.6486, 73.7647),
    "alandi road": (18.5612, 73.8767),
    "amnora park town": (18.5167, 73.9346),
    "aundh": (18.5619, 73.8102),
    "balewadi": (18.582, 73.769),
    "baner": (18.5642, 73.7769),
    "baner road": (18.5647, 73.7748),
    "bavdhan": (18.521, 73.7781),
    "bhosari": (18.621, 73.8501),
    "bhugaon": (18.5, 73.7501),
    "bibvewadi": (18.4782, 73.8621),
    "bt kawade road": (18.5467, 73.8800),
    "budhwar peth": (18.5176, 73.858),
    "bund garden": (18.5406, 73.8834),
    "camp": (18.5216, 73.8718),
    "chakan": (18.7623, 73.8625),
    "chandan nagar": (18.5571, 73.9281),
    "chikhali": (18.6642, 73.8267),
    "dapodi": (18.5817, 73.8327),
    "deccan gymkhana": (18.5159, 73.8412),
    "dhankawadi": (18.4653, 73.855),
    "dhanori": (18.5907, 73.8913),
    "dhayari": (18.4374, 73.819),
    "dhole patil road": (18.5349, 73.8767),
    "erandwane": (18.5087, 73.8318),
    "fc road": (18.515, 73.8422),
    "ganga dham": (18.4793, 73.8754),
    "gultekdi": (18.4939, 73.8676),
    "hinjewadi": (18.592, 73.7579),
    "hadapsar": (18.5089, 73.9260),
    "jm road": (18.525, 73.8496),
    "junnar": (19.2007, 73.9768),
    "kalyani nagar": (18.5481, 73.9026),
    "karve nagar": (18.4894, 73.8213),
    "kasba peth": (18.5219, 73.8583),
    "katraj": (18.4537, 73.8563),
    "keshav nagar": (18.5323, 73.9384),
    "khadki": (18.5682, 73.8508),
    "kharadi": (18.5513, 73.9417),
    "kondhwa": (18.478, 73.8941),
    "koregaon park": (18.5366, 73.8933),
    "kothrud": (18.5071, 73.8051),
    "law college road": (18.5166, 73.8294),
    "lohegaon": (18.5804, 73.9182),
    "lonavala": (18.7504, 73.4069),
    "lulla nagar": (18.4590, 73.8966),
    "magarpatta": (18.5112, 73.9274),
    "magarpatta city": (18.5112, 73.9274),
    "mahalunge": (19.0931, 73.7527),
    "mangalvar peth": (18.5210, 73.8650),
    "manik bag": (18.5400, 73.8300),
    "market yard": (18.3367, 74.3814),
    "marunji": (18.6117, 73.7156),
    "model colony": (18.5314, 73.8375),
    "mohammadwadi": (18.4733, 73.9238),
    "mohamadwadi": (18.4733, 73.9238),
    "moshi": (18.6523, 73.8442),
    "mukund nagar": (18.4952, 73.8662),
    "mundhwa": (18.5343, 73.9298),
    "nagar road": (18.5629, 73.9301),
    "nanded city": (18.4597, 73.786),
    "narayan peth": (18.5156, 73.8511),
    "navi peth": (18.5093, 73.8441),
    "nibm": (18.4642, 73.9029),
    "nigdi-pradhikaran": (18.6559, 73.7681),
    "parvati gaon": (18.5018, 73.8636),
    "parvati paytha": (18.4985, 73.8497),
    "pashan": (18.5387, 73.7953),
    "paud road": (18.5076, 73.7792),
    "pimple nilakh": (18.5697, 73.7941),
    "pimple saudagar": (18.5982, 73.7978),
    "pimpri-chinchwad": (18.6279, 73.801),
    "porwal road": (18.6086, 73.9111),
    "prabhat road": (18.512, 73.8316),
    "rahatani": (18.6025, 73.796),
    "rasta peth": (18.5198, 73.865),
    "ravet": (18.6433, 73.7451),
    "sadashiv peth": (18.5108, 73.8502),
    "salunke vihar": (18.4590, 73.8966),
    "sasanenagar": (18.5167, 73.9346),
    "senapati bapat marg": (18.5242, 73.8297),
    "shaniwar peth": (18.5193, 73.8525),
    "shankar shet road": (18.4952, 73.8662),
    "shivajinagar": (18.5326, 73.8513),
    "shukrawar peth": (18.5114, 73.854),
    "sinhagad road": (18.5005, 73.8446),
    "sopan baug": (18.5142, 73.9014),
    "swargate": (18.4989, 73.8585),
    "thergaon": (18.6093, 73.7729),
    "tilak road": (18.5033, 73.8561),
    "undri": (18.4516, 73.8927),
    "vadgaon budruk": (18.4675, 73.8254),
    "viman nagar": (18.5704, 73.9133),
    "vishrantwadi": (18.5726, 73.8782),
    "wadgaon sheri": (18.5436, 73.9241),
    "wagholi": (18.5806, 73.9833),
    "wakad": (18.6022, 73.7644),
    "wanowrie": (18.4884, 73.8987),
    "wanwadi": (18.4884, 73.8987),
    "warje": (18.482, 73.8002),
    "yerwada": (18.5453, 73.8867),
}

# Common spelling/format aliases mapped to canonical LOCATION_COORDS keys.
LOCATION_ALIASES = {
    "amanora park town": "amnora park town",
    "shivaji nagar": "shivajinagar",
    "mangalwar peth": "mangalvar peth",
    "bibwewadi": "bibvewadi",
}


def validate_location_coordinates():
    """
    Validate LOCATION_COORDS structure and coordinate ranges.
    """
    invalid_keys = [key for key in LOCATION_COORDS.keys() if key != key.lower()]
    if invalid_keys:
        raise ValueError(f"LOCATION_COORDS contains non-lowercase keys: {sorted(invalid_keys)}")

    invalid_coords = []
    for key, coords in LOCATION_COORDS.items():
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            invalid_coords.append(key)
            continue
        lat, lon = coords
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            invalid_coords.append(key)

    if invalid_coords:
        raise ValueError(f"Invalid coordinates for locations: {sorted(invalid_coords)}")


# Validate coordinates once at initialization
validate_location_coordinates()


def convert_coordinates_to_location(latitude: float, longitude: float, max_distance_km: float = 10.0) -> str:
    """
    Convert latitude/longitude coordinates to the nearest known location name.
    Uses Haversine formula to calculate distance between coordinates.
    
    Args:
        latitude (float): User's latitude coordinate
        longitude (float): User's longitude coordinate
        max_distance_km (float): Maximum distance in km to consider a match (default: 10km)
    
    Returns:
        str: Nearest location name (lowercase) or None if no location within max_distance
    
    Example:
        >>> convert_coordinates_to_location(18.5642, 73.7769)
        'baner'
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    # Find nearest location
    min_distance = float('inf')
    nearest_location = None
    
    for location_name, (loc_lat, loc_lon) in LOCATION_COORDS.items():
        distance = haversine_distance(latitude, longitude, loc_lat, loc_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_location = location_name
    
    # Return location only if within max_distance threshold
    if min_distance <= max_distance_km:
        return nearest_location
    else:
        return None


def _canonical_location_key(location_name: str) -> str:
    """
    Normalize location text for tolerant region-name matching.
    """
    text = str(location_name or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def normalize_location_name(location_name: str) -> str:
    """
    Normalize user-provided region to a canonical location key when possible.
    """
    normalized = str(location_name or "").strip().lower()
    if not normalized:
        return ""

    if normalized in LOCATION_COORDS:
        return normalized

    if normalized in LOCATION_ALIASES:
        return LOCATION_ALIASES[normalized]

    canonical_input = _canonical_location_key(normalized)
    if not canonical_input:
        return normalized

    for key in LOCATION_COORDS.keys():
        if _canonical_location_key(key) == canonical_input:
            return key

    for alias, target in LOCATION_ALIASES.items():
        if _canonical_location_key(alias) == canonical_input:
            return target

    return normalized


def get_coordinates_for_location(location_name: str):
    """
    Resolve a region name to LOCATION_COORDS.

    Accepts direct key matches and tolerant canonical matches
    (e.g., spaces/hyphens/commas/punctuation differences).
    """
    normalized = normalize_location_name(location_name)
    if not normalized:
        return None

    direct = LOCATION_COORDS.get(normalized)
    if direct:
        return {
            "latitude": float(direct[0]),
            "longitude": float(direct[1]),
            "matched_location": normalized,
        }

    return None


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two coordinates in kilometers.
    """
    from math import radians, sin, cos, sqrt, atan2

    earth_radius_km = 6371.0

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius_km * c


def _get_ordered_nearby_locations(
    user_location: str,
    candidate_locations=None
) -> list:
    """
    Build ordered nearby locations:
    1) Existing NEARBY_MAP order (preserved)
    2) Remaining candidate locations sorted by Haversine distance from LOCATION_COORDS
    """
    base_location = str(user_location).strip().lower()
    ordered = [loc for loc in NEARBY_MAP.get(base_location, []) if loc != base_location]
    seen = set(ordered)

    base_coords = LOCATION_COORDS.get(base_location)
    if base_coords is None:
        return ordered

    if candidate_locations is None:
        location_pool = set(LOCATION_COORDS.keys())
    else:
        location_pool = {
            str(loc).strip().lower()
            for loc in candidate_locations
            if str(loc).strip()
        }

    location_pool.discard(base_location)

    distance_ranked = []
    for candidate in location_pool:
        if candidate in seen:
            continue
        candidate_coords = LOCATION_COORDS.get(candidate)
        if candidate_coords is None:
            continue
        distance_km = _haversine_distance_km(
            base_coords[0], base_coords[1], candidate_coords[0], candidate_coords[1]
        )
        distance_ranked.append((distance_km, candidate))

    distance_ranked.sort(key=lambda item: item[0])
    ordered.extend([candidate for _, candidate in distance_ranked])
    return ordered


def _convert_to_json_serializable(value):
    """
    Convert numpy/pandas types to native Python types for JSON serialization.
    
    Args:
        value: Value to convert
    
    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    
    # Convert numpy types to Python native types
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    
    return value


def _get_valid_contact_number(contact_number):
    """
    Validate and clean contact_number for JSON serialization.
    
    Args:
        contact_number: Contact number from dataframe (can be str, None, NaN)
    
    Returns:
        str: Valid contact number string, or None if empty/invalid
    """
    if contact_number is None:
        return None
    
    # Convert to string and strip whitespace
    contact_str = str(contact_number).strip()
    
    # Return None if empty or represents missing value
    if not contact_str or contact_str.lower() in ['nan', 'none', '']:
        return None
    
    return contact_str


def load_doctors_data():
    """
    Load and clean the doctors data from CSV file.
    
    Preprocesses doctors_master_sheet.csv with column renaming, data type conversions,
    and cleaning operations.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized column names and types.
        
    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If missing LOCATION_COORDS entries or invalid data.
    """
    global _cached_data
    
    # Return cached data if already loaded
    if _cached_data is not None:
        return _cached_data
    
    file_path = _resolve_data_file("doctors_master_sheet.csv")
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path.absolute()}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # 1. Rename columns to standardized names
        column_mapping = {
            'doctors_name': 'name',
            'doctors_location': 'location',
            'doctors_rating': 'rating_score',
            'doctors_number': 'contact_number',
            'doctors_cost': 'consultation_fee',
            'specialty': 'specialty'
        }
        df = df.rename(columns=column_mapping)
        
        # 2. Clean location column: remove trailing commas, strip whitespace, convert to lowercase
        df['location'] = df['location'].str.strip().str.rstrip(',').str.lower()
        
        # 3. Clean rating_score: handle "unavailable", remove '%' symbol, convert to float, divide by 100
        df['rating_score'] = df['rating_score'].replace('unavailable', pd.NA)
        df['rating_score'] = df['rating_score'].astype(str).str.replace('%', '', regex=False)
        df['rating_score'] = pd.to_numeric(df['rating_score'], errors='coerce') / 100
        df['rating_score'] = df['rating_score'].fillna(0.0)
        
        # 4. Clean consultation_fee: remove '₹', convert to integer
        df['consultation_fee'] = df['consultation_fee'].astype(str).str.replace('₹', '', regex=False)
        df['consultation_fee'] = pd.to_numeric(df['consultation_fee'], errors='coerce').astype('Int64')
        
        # 5. Clean contact_number: handle "unavailable", convert to string, remove scientific notation and decimals
        df['contact_number'] = df['contact_number'].replace('unavailable', pd.NA)
        df['contact_number'] = df['contact_number'].astype(str)
        # Remove .0 suffix if present
        df['contact_number'] = df['contact_number'].str.replace(r'\.0$', '', regex=True)
        # Replace 'nan' string with 'unavailable'
        df.loc[df['contact_number'].str.lower() == 'nan', 'contact_number'] = 'unavailable'
        
        # 6. Convert specialty to lowercase
        df['specialty'] = df['specialty'].str.lower()
        
        # 7. Drop rows where name, specialty, or location are missing
        df = df.dropna(subset=['name', 'specialty', 'location'])
        
        # Select only standardized columns (remove profession duplicate)
        df = df[['name', 'specialty', 'location', 'consultation_fee', 'rating_score', 'contact_number']]
        
        # 8. Fill missing consultation fees with 0 (for doctors with unavailable fees)
        df['consultation_fee'] = df['consultation_fee'].fillna(0)
        
        # Ensure data types
        df['consultation_fee'] = df['consultation_fee'].astype('int')
        df['rating_score'] = df['rating_score'].astype('float')
        df['contact_number'] = df['contact_number'].astype('str')
        df['location'] = df['location'].astype('str')
        df['specialty'] = df['specialty'].astype('str')

        # Drop exact duplicate doctor rows, keeping the highest-rated entry.
        df['_name_key'] = df['name'].astype(str).str.strip().str.lower()
        df = df.sort_values(
            by=['_name_key', 'location', 'specialty', 'rating_score', 'consultation_fee'],
            ascending=[True, True, True, False, True],
        )
        df = df.drop_duplicates(subset=['_name_key', 'location', 'specialty'], keep='first')
        df = df.drop(columns=['_name_key']).reset_index(drop=True)

        # Validate LOCATION_COORDS covers all dataset locations
        dataset_locations = set(df['location'].dropna().unique())
        coords_locations = set(LOCATION_COORDS.keys())
        missing_locations = sorted(dataset_locations - coords_locations)
        if missing_locations:
            raise ValueError(
                f"Missing LOCATION_COORDS for locations: {missing_locations}"
            )
        
        # Cache the cleaned data
        _cached_data = df
        
        return _cached_data
    
    except Exception as e:
        raise Exception(f"Error loading doctors data: {str(e)}")


def _compute_cost_score(fees):
    """
    Compute cost score from consultation fees.
    Lower fees receive higher scores.
    
    Args:
        fees (pd.Series): Series of consultation fees.
    
    Returns:
        pd.Series: Cost scores normalized between 0 and 1.
    """
    min_fee = fees.min()
    max_fee = fees.max()
    
    if max_fee == min_fee:
        return pd.Series(1.0, index=fees.index)
    
    return 1 - (fees - min_fee) / (max_fee - min_fee)


def _compute_location_score(locations, user_location):
    """
    Compute location score based on matching user's preferred location.
    
    Args:
        locations (pd.Series): Series of doctor locations.
        user_location (str): User's preferred location.
    
    Returns:
        pd.Series: Location scores (1.0 for match, 0.5 otherwise).
    """
    user_location_key = str(user_location).strip().lower()
    user_coords = LOCATION_COORDS.get(user_location_key)

    # Keep legacy behavior when user location has no known coordinates.
    if user_coords is None:
        return locations.str.lower().eq(user_location_key).astype(float).where(
            locations.str.lower().eq(user_location_key), 0.5
        )

    def score_from_distance(location_value):
        location_key = str(location_value).strip().lower()
        if location_key == user_location_key:
            return 1.0

        candidate_coords = LOCATION_COORDS.get(location_key)
        if candidate_coords is None:
            return 0.5

        distance_km = _haversine_distance_km(
            user_coords[0], user_coords[1], candidate_coords[0], candidate_coords[1]
        )
        # Smooth decay: nearest regions get higher score, far regions converge lower.
        return max(0.2, 1.0 / (1.0 + distance_km / 5.0))

    return locations.apply(score_from_distance)


def _generate_reason(rating_score, consultation_fee, location_score, percentile_25):
    """
    Generate a reason/explanation string for a doctor recommendation.
    
    Args:
        rating_score (float): Doctor's rating score (0-1)
        consultation_fee (float): Doctor's consultation fee
        location_score (float): Location match score (1.0 or 0.5)
        percentile_25 (float): 25th percentile of consultation fees in filtered group
    
    Returns:
        str: Reason string combining applicable tags
    """
    reasons = []
    
    # Check if highly rated
    if rating_score >= 0.9:
        reasons.append("Highly rated")
    
    # Check if affordable (within bottom 25%)
    if consultation_fee <= percentile_25:
        reasons.append("Affordable")
    
    # Check if nearby
    if location_score == 1.0:
        reasons.append("Nearby")
    
    # Return combined reason or default
    return " • ".join(reasons) if reasons else "Balanced recommendation"


def rank_doctors(df, user_location, severity="medium", top_k=5, preferred_specialty=""):
    """
    Rank doctors based on a weighted scoring system adjusted by severity level.
    
    Args:
        df (pd.DataFrame): DataFrame containing doctor data with columns:
            name, specialty, location, consultation_fee, rating_score, contact_number.
        user_location (str): User's preferred location.
        severity (str): Severity level - "low", "medium", "high". Default is "medium".
        top_k (int): Number of top results to return. Default is 5.
        preferred_specialty (str): Specialty used to pick one row when aliases collide.
    
    Severity-based weighting:
        - "low": Rating 0.5, Cost 0.3, Location 0.2 (standard)
        - "medium": Rating 0.5, Cost 0.3, Location 0.2 (standard)
        - "high": Rating 0.6, Cost 0.2, Location 0.2 (prioritizes quality)
    
    Returns:
        list: List of dictionaries with top_k doctors sorted by final_score.
              Each dict contains: name, specialty, location, 
              consultation_fee, rating_score, final_score, reason, contact_number, latitude, longitude.
    """
    # Validate required columns
    required_columns = {'name', 'specialty', 'location', 'consultation_fee', 'rating_score', 'contact_number'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Create a copy to avoid modifying original data
    df = _dedupe_doctors_dataframe(df.copy(), preferred_specialty=preferred_specialty)
    if df.empty:
        return []
    
    user_location_key = str(user_location).strip().lower()

    # Compute scores
    df['cost_score'] = _compute_cost_score(df['consultation_fee'])
    df['location_score'] = _compute_location_score(df['location'], user_location)
    df['exact_location_match'] = df['location'].str.lower().eq(user_location_key)
    
    # Determine weights based on severity
    if severity == "high":
        rating_weight = 0.6
        cost_weight = 0.2
        location_weight = 0.2
    else:  # "low" and "medium" use same weights
        rating_weight = 0.5
        cost_weight = 0.3
        location_weight = 0.2
    
    # Compute final score with severity-adjusted weights
    df['final_score'] = (
        rating_weight * df['rating_score'] +
        cost_weight * df['cost_score'] +
        location_weight * df['location_score']
    )
    
    # Calculate 25th percentile of consultation fees for "Affordable" tag
    percentile_25 = df['consultation_fee'].quantile(0.25)
    
    # Prioritize exact-location matches as most relevant, then score quality.
    top_doctors = df.sort_values(
        by=['exact_location_match', 'final_score', 'rating_score', 'location_score', 'cost_score'],
        ascending=[False, False, False, False, False]
    ).head(top_k)
    
    # Format results as list of dictionaries
    results = []
    for _, row in top_doctors.iterrows():
        location_key = str(row['location']).strip().lower()
        coords = LOCATION_COORDS.get(location_key)
        latitude = float(coords[0]) if coords else None
        longitude = float(coords[1]) if coords else None
        # Generate reason string
        reason = _generate_reason(
            rating_score=float(_convert_to_json_serializable(row['rating_score'])),
            consultation_fee=float(_convert_to_json_serializable(row['consultation_fee'])),
            location_score=float(_convert_to_json_serializable(row['location_score'])),
            percentile_25=float(percentile_25)
        )
        
        results.append({
            'name': row['name'],
            'specialty': _display_specialty(row['specialty']),
            'location': row['location'],
            'consultation_fee': int(_convert_to_json_serializable(row['consultation_fee'])),
            # Expose doctor rating on a user-friendly 0-10 scale.
            'rating_score': round(float(_convert_to_json_serializable(row['rating_score'])) * 10, 1),
            'final_score': round(float(_convert_to_json_serializable(row['final_score'])), 4),
            'reason': reason,
            'contact_number': _get_valid_contact_number(row['contact_number']),
            'latitude': latitude,
            'longitude': longitude
        })
    
    return _dedupe_doctor_dicts(results, preferred_specialty=preferred_specialty)


def recommend_doctors(input_data: dict):
    """
    Recommend doctors based on specialty, location, and severity preferences.
    
    Fallback logic:
    1. Try exact specialty + exact location
    2. Try exact specialty + nearby locations (via NEARBY_MAP)
    3. Try general_physician + exact location
    4. Try general_physician + nearby locations
    5. If specialty results exist but are limited, supplement with general_physician
       using the same exact-location-then-nearby ordering.
    
    Args:
        input_data (dict): Dictionary containing:
            - "specialty" (str): Desired medical specialty (required)
            - "location" (str): Preferred location (required)
            - "severity" (str): Severity level - "low", "medium", "high" (optional, default: "medium")
    
    Returns:
        dict: Dictionary with keys:
            - "recommended_doctors": List of recommended doctor dicts
            - "total_matches": Total doctors matching the specialty
            - "metadata": Fallback information
            - "message": Optional message if no matches found
    """
    # Extract parameters
    specialty_input = input_data.get("specialty", "")
    specialty = canonicalize_specialty(specialty_input)
    location_input = input_data.get("location", "")
    severity = input_data.get("severity", "medium").strip().lower()

    # Support both location-name and coordinates input.
    if isinstance(location_input, dict):
        latitude = location_input.get("latitude")
        longitude = location_input.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("Location coordinates must include both latitude and longitude.")

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError):
            raise ValueError("Invalid latitude or longitude values.")

        detected_location = convert_coordinates_to_location(latitude, longitude)
        if not detected_location:
            raise ValueError("No known region found near the provided coordinates.")
        location = detected_location
    else:
        location = normalize_location_name(location_input)
    
    # Validate inputs
    if not specialty or not location:
        raise ValueError("Both 'specialty' and 'location' are required.")
    
    # Load dataset
    df = load_doctors_data()
    
    df = df.copy()
    df["specialty_canonical"] = df["specialty"].apply(canonicalize_specialty)
    filtered_df = df[df["specialty_canonical"] == specialty]

    fallback_applied = False
    fallback_location = None
    fallback_type = None
    original_specialty = specialty
    supplemented_with_general_physician = False

    if filtered_df.empty:
        # No exact specialty match - this could be valid if it genuinely doesn't exist
        # Don't auto-fallback here, let the location filtering handle it
        pass
    else:
        # First attempt: exact location match
        exact_location_filtered = filtered_df[filtered_df['location'].str.lower() == location]
        location_filtered = exact_location_filtered.copy()

        # If exact-location matches are missing or limited, top up from nearby locations.
        if len(location_filtered) < DEFAULT_TOP_K:
            available_specialty_locations = set(filtered_df['location'].str.lower().unique())
            nearby_locations = _get_ordered_nearby_locations(location, available_specialty_locations)
            nearby_batches = []
            collected_count = len(location_filtered)

            for nearby in nearby_locations:
                nearby_filtered = filtered_df[filtered_df['location'].str.lower() == nearby]
                if nearby_filtered.empty:
                    continue

                needed_count = DEFAULT_TOP_K - collected_count
                nearby_limited = nearby_filtered.head(needed_count)
                nearby_batches.append(nearby_limited)
                collected_count += len(nearby_limited)

                fallback_applied = True
                if fallback_location is None:
                    fallback_location = nearby
                fallback_type = "nearby"

                if collected_count >= DEFAULT_TOP_K:
                    break

            if nearby_batches:
                location_filtered = pd.concat([location_filtered] + nearby_batches, ignore_index=True)

        filtered_df = location_filtered
    
    # If still no matches, try general_physician fallback
    if filtered_df.empty:
        gp_df = df[df["specialty_canonical"] == "general physician"]

        # Try exact location first
        gp_exact_filtered = gp_df[gp_df['location'].str.lower() == location]
        gp_location_filtered = gp_exact_filtered.copy()

        # If exact GP matches are missing or limited, top up from nearby locations.
        if len(gp_location_filtered) < DEFAULT_TOP_K:
            available_gp_locations = set(gp_df['location'].str.lower().unique())
            nearby_locations = _get_ordered_nearby_locations(location, available_gp_locations)
            nearby_gp_batches = []
            collected_gp_count = len(gp_location_filtered)

            for nearby in nearby_locations:
                nearby_gp = gp_df[gp_df['location'].str.lower() == nearby]
                if nearby_gp.empty:
                    continue

                needed_gp_count = DEFAULT_TOP_K - collected_gp_count
                nearby_gp_limited = nearby_gp.head(needed_gp_count)
                nearby_gp_batches.append(nearby_gp_limited)
                collected_gp_count += len(nearby_gp_limited)

                fallback_applied = True
                if fallback_location is None:
                    fallback_location = nearby
                fallback_type = "general_physician_nearby"

                if collected_gp_count >= DEFAULT_TOP_K:
                    break

            if nearby_gp_batches:
                gp_location_filtered = pd.concat([gp_location_filtered] + nearby_gp_batches, ignore_index=True)

        if not gp_location_filtered.empty:
            filtered_df = gp_location_filtered
            fallback_applied = True
            if fallback_type != "general_physician_nearby":
                fallback_type = "general_physician"
    # If specialty matches are available but limited, supplement with general physicians
    # using the same distance logic (exact location first, then nearby in order).
    elif len(filtered_df) < DEFAULT_TOP_K and specialty != "general physician":
        gp_df = df[df["specialty_canonical"] == "general physician"]
        needed_count = DEFAULT_TOP_K - len(filtered_df)
        gp_batches = []
        gp_nearby_used = False

        # Try exact location first
        gp_exact = gp_df[gp_df['location'].str.lower() == location]
        if not gp_exact.empty and needed_count > 0:
            gp_exact_limited = gp_exact.head(needed_count)
            gp_batches.append(gp_exact_limited)
            needed_count -= len(gp_exact_limited)
            fallback_applied = True
            fallback_type = "general_physician"

        # Then try nearby locations in order
        if needed_count > 0:
            available_gp_locations = set(gp_df['location'].str.lower().unique())
            nearby_locations = _get_ordered_nearby_locations(location, available_gp_locations)
            for nearby in nearby_locations:
                nearby_gp = gp_df[gp_df['location'].str.lower() == nearby]
                if nearby_gp.empty:
                    continue

                nearby_gp_limited = nearby_gp.head(needed_count)
                gp_batches.append(nearby_gp_limited)
                needed_count -= len(nearby_gp_limited)
                gp_nearby_used = True
                fallback_applied = True

                if fallback_location is None:
                    fallback_location = nearby

                if needed_count <= 0:
                    break

        if gp_batches:
            filtered_df = pd.concat([filtered_df] + gp_batches, ignore_index=True)
            supplemented_with_general_physician = True
            if gp_nearby_used:
                fallback_type = "general_physician_nearby"
    
    # Handle no matches
    if filtered_df.empty:
        return {
            "recommended_doctors": [],
            "total_matches": 0,
            "message": "No doctors found for this specialty or fallback.",
            "metadata": {
                "fallback_applied": False,
                "fallback_location": None,
                "fallback_type": None,
                "original_specialty": original_specialty
            }
        }
    
    # Get total matches before trimming
    total_matches = len(filtered_df)
    
    # Rank doctors based on severity
    recommended = rank_doctors(
        filtered_df,
        location,
        severity=severity,
        top_k=DEFAULT_TOP_K,
        preferred_specialty=specialty,
    )

    # Keep requested specialty doctors first, then GP supplements.
    if supplemented_with_general_physician:
        specialty_first = [
            doctor for doctor in recommended
            if str(doctor.get("specialty", "")).lower() != "general physician"
        ]
        gp_supplements = [
            doctor for doctor in recommended
            if str(doctor.get("specialty", "")).lower() == "general physician"
        ]
        recommended = specialty_first + gp_supplements

    recommended = _dedupe_doctor_dicts(recommended, preferred_specialty=specialty)
    
    # Return structured response
    return {
        "recommended_doctors": recommended,
        "total_matches": total_matches,
        "metadata": {
            "fallback_applied": fallback_applied,
            "fallback_location": fallback_location,
            "fallback_type": fallback_type,
            "original_specialty": original_specialty,
            "supplemented_with_general_physician": supplemented_with_general_physician
        }
    }


def _map_specialty_to_hospital_format(specialty: str) -> list:
    """
    Map doctor specialty names to hospital specialty names.
    Handles conversion like 'cardiologist' -> 'cardiology'.
    
    Args:
        specialty (str): Doctor specialty name (lowercase)
    
    Returns:
        list: List of possible hospital specialty names to search for
    """
    # Mapping from doctor specialties to hospital specialties
    specialty_map = {
        'cardiologist': ['cardiology'],
        'dentist': ['dentistry', 'dental'],
        'dentists': ['dentistry', 'dental'],
        'dermatologist': ['dermatology'],
        'endocrinologist': ['endocrinology'],
        'ent specialist': ['ent', 'ear nose throat'],
        'gastroenterologist': ['gastroenterology'],
        'general physician': ['general physician', 'general medicine'],
        'gynac': ['gynecology', 'obstetrics and gynaecology'],
        'internal medicine': ['internal medicine'],
        'nephrologist': ['nephrology'],
        'obsstetrician': ['obstetrics'],
        'oncologist': ['oncology'],
        'ophthalmologist': ['ophthalmology', 'eye'],
        'orthopedic': ['orthopedic', 'orthopedics'],
        'pediatrician': ['pediatrics', 'paediatrics'],
        'psychiatrist': ['psychiatry'],
        'pulmonologist': ['pulmonology', 'respiratory'],
        'urologist': ['urology'],
    }
    
    # Get mapped specialties, or use original if no mapping found
    mapped = specialty_map.get(specialty, [specialty])
    return mapped


def generate_cost_insights(input_data: dict):
    """
    Generate cost insights for doctors matching specialty and location.
    
    Args:
        input_data (dict): Dictionary containing:
            - "specialty" (str): Desired medical specialty
            - "location" (str): Preferred location
    
    Returns:
        dict: Dictionary with keys:
            - "average_fee": Average consultation fee (rounded to 2 decimals)
            - "min_fee": Minimum consultation fee
            - "max_fee": Maximum consultation fee
            - "cost_band": Cost category ("Budget-friendly", "Standard", or "Premium")
            - "message": Optional message if no data found
    """
    # Extract parameters
    specialty = input_data.get("specialty", "").strip()
    location = input_data.get("location", "").strip()
    
    # Validate inputs
    if not specialty or not location:
        raise ValueError("Both 'specialty' and 'location' are required.")
    
    # Load dataset
    df = load_doctors_data()
    
    # Filter by specialty (case insensitive)
    filtered_df = df[df['specialty'].str.lower() == specialty.lower()]
    
    # Further filter by location (case insensitive)
    filtered_df = filtered_df[filtered_df['location'].str.lower() == location.lower()]
    
    # Handle no matches
    if filtered_df.empty:
        return {
            "average_fee": None,
            "min_fee": None,
            "max_fee": None,
            "cost_band": None,
            "message": "No cost data available for this selection."
        }
    
    # Compute fee statistics and convert to native Python types
    average_fee = float(round(float(_convert_to_json_serializable(filtered_df['consultation_fee'].mean())), 2))
    min_fee = float(round(float(_convert_to_json_serializable(filtered_df['consultation_fee'].min())), 2))
    max_fee = float(round(float(_convert_to_json_serializable(filtered_df['consultation_fee'].max())), 2))
    
    # Determine cost band
    if average_fee < 300:
        cost_band = "Budget-friendly"
    elif average_fee <= 500:
        cost_band = "Standard"
    else:
        cost_band = "Premium"
    
    # Return structured response
    return {
        "average_fee": average_fee,
        "min_fee": min_fee,
        "max_fee": max_fee,
        "cost_band": cost_band
    }


def _generate_cost_summary_from_recommendations(recommended_doctors: list) -> dict:
    """
    Generate cost summary directly from returned recommended doctors.

    Args:
        recommended_doctors (list): List of doctor recommendation dicts.

    Returns:
        dict: Dictionary with average_fee, min_fee, max_fee, and cost_band.
    """
    if not recommended_doctors:
        return {
            "average_fee": None,
            "min_fee": None,
            "max_fee": None,
            "cost_band": None
        }

    fees = []
    for doctor in recommended_doctors:
        fee = doctor.get("consultation_fee")
        if fee is None:
            continue
        try:
            fees.append(float(fee))
        except (TypeError, ValueError):
            continue

    if not fees:
        return {
            "average_fee": None,
            "min_fee": None,
            "max_fee": None,
            "cost_band": None
        }

    average_fee = round(sum(fees) / len(fees), 2)
    min_fee = min(fees)
    max_fee = max(fees)

    if average_fee < 300:
        cost_band = "Budget-friendly"
    elif average_fee <= 500:
        cost_band = "Standard"
    else:
        cost_band = "Premium"

    return {
        "average_fee": average_fee,
        "min_fee": min_fee,
        "max_fee": max_fee,
        "cost_band": cost_band
    }


def hospital_recommendation_engine(location: str, specialty: str = None):
    """
    Recommend hospitals based on location and optional specialty requirement.
    
    Args:
        location (str): Preferred location (required, case-insensitive)
        specialty (str): Optional medical specialty to filter hospitals by capabilities
    
    Returns:
        dict: Dictionary with keys:
            - "recommended_hospitals": List of hospital dicts
            - "total_matches": Total hospitals found
            - "message": Optional message if no hospitals found
    
    Hospital dict contains:
        - hospital_name, location, hospital_type, emergency_available, 
          icu_available, specialties_available, latitude, longitude, contact_number
    """
    try:
        file_path = _resolve_data_file("hospitals.csv")
        
        if not file_path.exists():
            return {
                "recommended_hospitals": [],
                "total_matches": 0,
                "message": "Hospital data not available."
            }
        
        # Load hospitals data
        hospitals_df = pd.read_csv(file_path)
        
        # Standardize location to lowercase
        location = location.strip().lower()
        
        # Filter by location (case insensitive)
        location_filtered = hospitals_df[hospitals_df['location'].str.lower() == location]
        
        if location_filtered.empty:
            return {
                "recommended_hospitals": [],
                "total_matches": 0,
                "message": f"No hospitals found in {location}."
            }
        
        # Optional: filter by specialty if provided
        if specialty:
            specialty = specialty.strip().lower()
            # Filter hospitals that have the specialty in their specialties_available
            location_filtered = location_filtered[
                location_filtered['specialties_available'].str.lower().str.contains(specialty, na=False)
            ]
        
        if location_filtered.empty:
            return {
                "recommended_hospitals": [],
                "total_matches": 0,
                "message": f"No hospitals found in {location} with {specialty} specialty."
            }
        
        # Format results
        results = []
        for _, row in location_filtered.iterrows():
            results.append({
                'hospital_name': row['hospital_name'],
                'location': row['location'],
                'hospital_type': row['hospital_type'],
                'emergency_available': bool(row['emergency_available']),
                'icu_available': bool(row['icu_available']),
                'specialties_available': row['specialties_available'],
                'contact_number': _get_valid_contact_number(row['contact_number']),
                'latitude': float(_convert_to_json_serializable(row['latitude'])),
                'longitude': float(_convert_to_json_serializable(row['longitude']))
            })
        
        return {
            "recommended_hospitals": results,
            "total_matches": len(results)
        }
    
    except Exception as e:
        return {
            "recommended_hospitals": [],
            "total_matches": 0,
            "message": f"Error retrieving hospital data: {str(e)}"
        }


def _normalize_boolean_flag(value) -> bool:
    """
    Normalize CSV boolean-like values to strict bool.
    """
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _prepare_hospitals_dataframe(hospitals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns used by hospital fallback/ranking logic.
    """
    prepared = hospitals_df.copy()
    prepared["location_lower"] = prepared["location"].astype(str).str.strip().str.lower()
    prepared["emergency_available_bool"] = prepared["emergency_available"].apply(_normalize_boolean_flag)
    prepared["icu_available_bool"] = prepared["icu_available"].apply(_normalize_boolean_flag)
    return prepared


def _dedupe_preserve_order(values: list) -> list:
    """
    Deduplicate list entries while preserving input order.
    """
    seen = set()
    ordered = []
    for value in values:
        key = str(value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _rank_and_format_hospital_results(
    hospitals_df: pd.DataFrame,
    requested_location: str,
    ordered_locations: list = None,
    top_k: int = DEFAULT_TOP_K
) -> tuple[list, int]:
    """
    Rank and format hospital results using existing score model with optional
    location-order priority for fallback scenarios.
    """
    if hospitals_df is None or hospitals_df.empty:
        return [], 0

    ranked_df = hospitals_df.copy()
    requested_location = str(requested_location or "").strip().lower()

    location_rank = {}
    if ordered_locations:
        location_rank = {loc: idx for idx, loc in enumerate(_dedupe_preserve_order(ordered_locations))}

    if location_rank:
        def _location_score_for(loc):
            if loc == requested_location:
                return 1.0
            rank = location_rank.get(loc)
            if rank is None:
                return 0.4
            return max(0.4, 0.95 - 0.04 * min(rank, 12))
    else:
        def _location_score_for(loc):
            return 1.0 if loc == requested_location else 0.7

    ranked_df["location_score"] = ranked_df["location_lower"].apply(_location_score_for)

    def get_hospital_type_score(h_type):
        h_type_lower = str(h_type).lower() if h_type else ""
        if "private" in h_type_lower:
            return 1.0
        if "public" in h_type_lower or "government" in h_type_lower:
            return 0.8
        return 0.6

    ranked_df["hospital_type_score"] = ranked_df["hospital_type"].apply(get_hospital_type_score)
    ranked_df["emergency_score"] = ranked_df["emergency_available_bool"].apply(lambda x: 0.3 if x else 0.0)
    ranked_df["icu_score"] = ranked_df["icu_available_bool"].apply(lambda x: 0.2 if x else 0.0)
    ranked_df["score"] = (
        0.5 * ranked_df["location_score"] +
        0.3 * ranked_df["hospital_type_score"] +
        0.1 * ranked_df["emergency_score"] +
        0.1 * ranked_df["icu_score"]
    )

    if location_rank:
        ranked_df["location_rank"] = ranked_df["location_lower"].apply(lambda loc: location_rank.get(loc, 9999))
        top_hospitals = ranked_df.sort_values(
            by=["score", "location_rank", "emergency_available_bool", "icu_available_bool"],
            ascending=[False, True, False, False]
        ).head(top_k)
    else:
        top_hospitals = ranked_df.nlargest(top_k, "score")

    results = []
    for _, row in top_hospitals.iterrows():
        results.append({
            "hospital_name": row["hospital_name"],
            "location": row["location"],
            "hospital_type": row["hospital_type"],
            "emergency_available": bool(row["emergency_available_bool"]),
            "icu_available": bool(row["icu_available_bool"]),
            "specialties_available": row["specialties_available"],
            "contact_number": _get_valid_contact_number(row["contact_number"]),
            "latitude": float(_convert_to_json_serializable(row["latitude"])),
            "longitude": float(_convert_to_json_serializable(row["longitude"])),
            "score": round(float(_convert_to_json_serializable(row["score"])), 4)
        })

    results = _dedupe_hospital_records_by_brand(results, top_k=top_k)
    return results, len(ranked_df)


def _build_hospital_response_with_metadata(
    hospitals_df: pd.DataFrame,
    requested_location: str,
    ordered_locations: list,
    metadata: dict
) -> dict:
    """
    Build a standardized hospital response payload from a dataframe.
    """
    recommended_hospitals, total_matches = _rank_and_format_hospital_results(
        hospitals_df,
        requested_location=requested_location,
        ordered_locations=ordered_locations
    )

    return {
        "care_setting": "hospital",
        "recommended_hospitals": recommended_hospitals,
        "total_matches": total_matches,
        "metadata": metadata
    }


def _final_area_wise_hospital_candidates(
    hospitals_df: pd.DataFrame,
    ordered_locations: list
) -> pd.DataFrame:
    """
    Build nearest area-wise hospital candidates (one preferred hospital per area),
    then top up with remaining nearest hospitals if needed.
    """
    if hospitals_df.empty:
        return hospitals_df

    selected_batches = []
    selected_indices = set()

    for loc in ordered_locations:
        loc_df = hospitals_df[hospitals_df["location_lower"] == loc]
        if loc_df.empty:
            continue

        # Prefer emergency/ICU-capable hospital for each location.
        loc_best = loc_df.sort_values(
            by=["emergency_available_bool", "icu_available_bool"],
            ascending=[False, False]
        ).head(1)
        selected_batches.append(loc_best)
        selected_indices.update(loc_best.index.tolist())

        if len(selected_batches) >= DEFAULT_TOP_K:
            break

    if selected_batches:
        selected_df = pd.concat(selected_batches, ignore_index=False)
    else:
        selected_df = hospitals_df.head(0)

    needed = DEFAULT_TOP_K - len(selected_df)
    if needed > 0:
        remaining_df = hospitals_df[~hospitals_df.index.isin(selected_indices)].copy()
        if not remaining_df.empty:
            location_rank = {loc: idx for idx, loc in enumerate(ordered_locations)}
            remaining_df["location_rank"] = remaining_df["location_lower"].apply(
                lambda loc: location_rank.get(loc, 9999)
            )
            top_up = remaining_df.sort_values(
                by=["location_rank", "emergency_available_bool", "icu_available_bool"],
                ascending=[True, False, False]
            ).head(needed)
            if not top_up.empty:
                selected_df = pd.concat([selected_df, top_up], ignore_index=False)

    return selected_df


def recommend_hospitals(input_data: dict):
    """
    Recommend hospitals based on location, specialty, and emergency requirements.
    
    Filtering logic:
    1. Load hospital dataset
    2. Filter by location (exact match first, then NEARBY_MAP fallback)
    3. Filter by emergency_available == True for high severity
    4. Filter by specialty (must be in specialties_available list)
    5. Rank hospitals by location proximity (exact match first) and hospital_type preference
    6. Return top DEFAULT_TOP_K hospitals
    
    Args:
        input_data (dict): Dictionary containing:
            - "location" (str): Preferred location (required)
            - "specialty" (str): Medical specialty (required)
            - "severity" (str): Severity level - "low", "medium", "high" (optional, default: "medium")
    
    Returns:
        dict: Dictionary with keys:
            - "care_setting": "hospital"
            - "recommended_hospitals": List of hospital dicts sorted by score
            - "total_matches": Total hospitals found
            - "metadata": Fallback and filtering information
            - "message": Optional message if no hospitals found
    
    Hospital dict contains:
        - hospital_name, location, hospital_type, emergency_available, 
          icu_available, specialties_available, latitude, longitude, contact_number, score
    """
    try:
        # Extract parameters
        location = input_data.get("location", "").strip().lower()
        specialty = canonicalize_specialty(input_data.get("specialty", ""))
        severity = input_data.get("severity", "medium").strip().lower()
        
        # Validate inputs
        if not location or not specialty:
            raise ValueError("Both 'location' and 'specialty' are required.")
        
        file_path = _resolve_data_file("hospitals.csv")
        
        if not file_path.exists():
            return {
                "care_setting": "hospital",
                "recommended_hospitals": [],
                "total_matches": 0,
                "message": "Hospital data not available.",
                "metadata": {
                    "fallback_applied": False,
                    "fallback_location": None
                }
            }
        
        # Load hospitals data
        hospitals_df = pd.read_csv(file_path)
        
        # Convert location to lowercase for comparison
        hospitals_df['location_lower'] = hospitals_df['location'].str.lower()
        
        fallback_applied = False
        fallback_location = None
        
        # Step 1: Filter by location (exact match first, then nearby)
        location_filtered = hospitals_df[hospitals_df['location_lower'] == location]
        fallback_applied = False
        fallback_location = None
        
        # If no exact results, try nearby locations
        if location_filtered.empty:
            nearby_locations = NEARBY_MAP.get(location, [])
            for nearby in nearby_locations:
                nearby_filtered = hospitals_df[hospitals_df['location_lower'] == nearby]
                
                # If high severity, check if this nearby location has emergency hospitals
                if severity == "high":
                    nearby_emergency = nearby_filtered[nearby_filtered['emergency_available'].astype(bool)]
                    if not nearby_emergency.empty:
                        location_filtered = nearby_emergency
                        fallback_applied = True
                        fallback_location = nearby
                        break
                else:
                    # For non-high severity, accept any hospitals
                    if not nearby_filtered.empty:
                        location_filtered = nearby_filtered
                        fallback_applied = True
                        fallback_location = nearby
                        break
        
        if location_filtered.empty:
            # Try one more time: for high severity, check if there are ANY emergency hospitals
            # in nearby locations (not just the specialty filter)
            if severity == "high":
                return {
                    "care_setting": "hospital",
                    "recommended_hospitals": [],
                    "total_matches": 0,
                    "message": "No emergency-capable hospitals found in this area or nearby locations.",
                    "metadata": {
                        "fallback_applied": False,
                        "fallback_location": None
                    }
                }
            else:
                return {
                    "care_setting": "hospital",
                    "recommended_hospitals": [],
                    "total_matches": 0,
                    "message": f"No hospitals found in {location} or nearby locations.",
                    "metadata": {
                        "fallback_applied": False,
                        "fallback_location": None
                    }
                }
        
        # Step 2: Filter by emergency_available if high severity (for exact location match)
        # For nearby locations, we've already applied this filter above
        if severity == "high" and not fallback_applied:
            location_filtered = location_filtered[location_filtered['emergency_available'].astype(bool)]
            
            if location_filtered.empty:
                # No emergency hospitals in exact location, try nearby
                nearby_locations = NEARBY_MAP.get(location, [])
                for nearby in nearby_locations:
                    nearby_filtered = hospitals_df[hospitals_df['location_lower'] == nearby]
                    nearby_emergency = nearby_filtered[nearby_filtered['emergency_available'].astype(bool)]
                    if not nearby_emergency.empty:
                        location_filtered = nearby_emergency
                        fallback_applied = True
                        fallback_location = nearby
                        break
                
                if location_filtered.empty:
                    return {
                        "care_setting": "hospital",
                        "recommended_hospitals": [],
                        "total_matches": 0,
                        "message": "No emergency-capable hospitals found in this area or nearby locations.",
                        "metadata": {
                            "fallback_applied": False,
                            "fallback_location": None
                        }
                    }
        
        # Step 3: Filter by specialty using canonical specialty keys.
        specialty_filtered = location_filtered[
            location_filtered["specialties_available"].apply(
                lambda value: _hospital_matches_specialty(value, specialty)
            )
        ]
        
        # If no hospitals with specialty found, allow any emergency hospitals for high severity
        # (for emergency cases, having an emergency-capable hospital is more important than specialty match)
        if specialty_filtered.empty and severity == "high":
            location_filtered = location_filtered  # Use location_filtered (which already has emergency hospitals)
        elif specialty_filtered.empty and fallback_applied and severity == "high":
            # No specialty match even in fallback location, use all emergency hospitals from fallback
            location_filtered = hospitals_df[hospitals_df['location_lower'] == fallback_location]
            location_filtered = location_filtered[location_filtered['emergency_available'].astype(bool)]
        else:
            location_filtered = specialty_filtered
        
        if location_filtered.empty:
            return {
                "care_setting": "hospital",
                "recommended_hospitals": [],
                "total_matches": 0,
                "message": f"No hospitals found with {specialty} specialty.",
                "metadata": {
                    "fallback_applied": fallback_applied,
                    "fallback_location": fallback_location
                }
            }
        
        # Get total matches before ranking
        total_matches = len(location_filtered)
        
        # Step 4: Rank hospitals by location proximity and hospital_type preference
        location_filtered = location_filtered.copy()
        
        # Score for location proximity (exact match = 1.0, nearby = 0.7)
        location_filtered['location_score'] = location_filtered['location_lower'].apply(
            lambda x: 1.0 if x == location else 0.7
        )
        
        # Score for hospital type (private = 1.0, public = 0.8, others = 0.6)
        def get_hospital_type_score(h_type):
            h_type_lower = str(h_type).lower() if h_type else ""
            if "private" in h_type_lower:
                return 1.0
            elif "public" in h_type_lower or "government" in h_type_lower:
                return 0.8
            else:
                return 0.6
        
        location_filtered['hospital_type_score'] = location_filtered['hospital_type'].apply(
            get_hospital_type_score
        )
        
        # Score for emergency services (true = 0.3 boost, false = 0)
        location_filtered['emergency_score'] = location_filtered['emergency_available'].astype(bool).apply(
            lambda x: 0.3 if x else 0.0
        )
        
        # Score for ICU availability (true = 0.2 boost, false = 0)
        location_filtered['icu_score'] = location_filtered['icu_available'].astype(bool).apply(
            lambda x: 0.2 if x else 0.0
        )
        
        # Compute final score: location (0.5) + hospital_type (0.3) + emergency (0.1) + icu (0.1)
        location_filtered['score'] = (
            0.5 * location_filtered['location_score'] +
            0.3 * location_filtered['hospital_type_score'] +
            0.1 * location_filtered['emergency_score'] +
            0.1 * location_filtered['icu_score']
        )
        
        # Step 5: Get top DEFAULT_TOP_K hospitals sorted by score descending
        top_hospitals = location_filtered.nlargest(DEFAULT_TOP_K, 'score')
        
        # Step 6: Format results
        results = []
        for _, row in top_hospitals.iterrows():
            results.append({
                'hospital_name': row['hospital_name'],
                'location': row['location'],
                'hospital_type': row['hospital_type'],
                'emergency_available': bool(row['emergency_available']),
                'icu_available': bool(row['icu_available']),
                'specialties_available': row['specialties_available'],
                'contact_number': _get_valid_contact_number(row['contact_number']),
                'latitude': float(_convert_to_json_serializable(row['latitude'])),
                'longitude': float(_convert_to_json_serializable(row['longitude'])),
                'score': round(float(_convert_to_json_serializable(row['score'])), 4)
            })

        results = _dedupe_hospital_records_by_brand(results, top_k=DEFAULT_TOP_K)

        # Return structured response
        return {
            "care_setting": "hospital",
            "recommended_hospitals": results,
            "total_matches": total_matches,
            "metadata": {
                "fallback_applied": fallback_applied,
                "fallback_location": fallback_location,
                "emergency_filter_applied": severity == "high",
                "specialty_requested": specialty
            }
        }
    
    except Exception as e:
        return {
            "care_setting": "hospital",
            "recommended_hospitals": [],
            "total_matches": 0,
            "message": f"Error processing hospital recommendations: {str(e)}",
            "metadata": {
                "fallback_applied": False,
                "fallback_location": None
            }
        }


def recommend_hospitals_with_guaranteed_fallback(input_data: dict):
    """
    Preserve current strict hospital logic as first pass, then progressively relax
    filters so medium/high flows can return hospitals whenever dataset entries exist.
    """
    # First pass always uses the existing strict logic unchanged.
    strict_results = recommend_hospitals(input_data)
    strict_count = strict_results.get("total_matches", 0)
    strict_metadata = strict_results.get("metadata", {}) or {}

    location = str(input_data.get("location", "")).strip().lower()
    specialty = canonicalize_specialty(input_data.get("specialty", ""))
    severity = str(input_data.get("severity", "medium")).strip().lower()
    strict_message = strict_results.get("message")

    if strict_count > 0:
        strict_results["metadata"] = {
            **strict_metadata,
            "strict_first_pass_applied": True,
            "strict_result_empty": False,
            "strict_message": strict_message,
            "guaranteed_fallback_applied": False,
            "guaranteed_fallback_stage": None,
            "strict_emergency_unavailable": False
        }
        return strict_results

    file_path = _resolve_data_file("hospitals.csv")
    if not file_path.exists():
        strict_results["metadata"] = {
            **strict_metadata,
            "strict_first_pass_applied": True,
            "strict_result_empty": True,
            "strict_message": strict_message,
            "guaranteed_fallback_applied": False,
            "guaranteed_fallback_stage": None,
            "strict_emergency_unavailable": severity == "high"
        }
        return strict_results

    try:
        hospitals_df = pd.read_csv(file_path)
    except Exception:
        strict_results["metadata"] = {
            **strict_metadata,
            "strict_first_pass_applied": True,
            "strict_result_empty": True,
            "strict_message": strict_message,
            "guaranteed_fallback_applied": False,
            "guaranteed_fallback_stage": None,
            "strict_emergency_unavailable": severity == "high"
        }
        return strict_results

    if hospitals_df.empty:
        strict_results["metadata"] = {
            **strict_metadata,
            "strict_first_pass_applied": True,
            "strict_result_empty": True,
            "strict_message": strict_message,
            "guaranteed_fallback_applied": False,
            "guaranteed_fallback_stage": None,
            "strict_emergency_unavailable": severity == "high"
        }
        return strict_results

    prepared_df = _prepare_hospitals_dataframe(hospitals_df)
    available_locations = set(prepared_df["location_lower"].dropna().astype(str).str.strip().str.lower().unique())

    nearby_seed = [loc for loc in NEARBY_MAP.get(location, []) if loc in available_locations]
    ordered_exact_nearby = _dedupe_preserve_order([location] + nearby_seed)
    scoped_df = prepared_df[prepared_df["location_lower"].isin(ordered_exact_nearby)].copy()

    def _apply_specialty_filter(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame[
            frame["specialties_available"].apply(
                lambda value: _hospital_matches_specialty(value, specialty)
            )
        ]

    staged_candidates = []

    # Stage 1: Expanded strict scope across exact + nearby locations.
    expanded_strict_df = _apply_specialty_filter(scoped_df)
    if severity == "high":
        expanded_strict_df = expanded_strict_df[expanded_strict_df["emergency_available_bool"]]
    staged_candidates.append((
        "expanded_strict_scope",
        expanded_strict_df,
        ordered_exact_nearby,
        "Expanded strict filtering across exact and nearby locations."
    ))

    # Stage 2: Relax specialty, keep emergency requirement (high) or keep all (medium).
    if severity == "high":
        relaxed_specialty_df = scoped_df[scoped_df["emergency_available_bool"]]
        staged_candidates.append((
            "relaxed_specialty_emergency_only",
            relaxed_specialty_df,
            ordered_exact_nearby,
            "Relaxed specialty filter while preserving emergency capability."
        ))
    else:
        staged_candidates.append((
            "relaxed_specialty",
            scoped_df,
            ordered_exact_nearby,
            "Relaxed specialty filter across exact and nearby locations."
        ))

    # Stage 3 (high only): Relax emergency requirement if still empty.
    if severity == "high":
        staged_candidates.append((
            "relaxed_emergency_and_specialty",
            scoped_df,
            ordered_exact_nearby,
            "Relaxed emergency and specialty filters across exact and nearby locations."
        ))

    # Stage 4: Final nearest area-wise hospitals across dataset (guaranteed fallback).
    ordered_all_locations = _dedupe_preserve_order(
        [location] + _get_ordered_nearby_locations(location, available_locations)
    )
    ordered_all_locations = [loc for loc in ordered_all_locations if loc in available_locations]
    if not ordered_all_locations:
        ordered_all_locations = sorted(available_locations)

    area_wise_df = _final_area_wise_hospital_candidates(prepared_df, ordered_all_locations)
    staged_candidates.append((
        "nearest_area_wise",
        area_wise_df,
        ordered_all_locations,
        "Nearest area-wise hospitals from dataset."
    ))

    for stage_name, stage_df, stage_locations, stage_reason in staged_candidates:
        if stage_df.empty:
            continue

        stage_locations_with_results = _dedupe_preserve_order(stage_df["location_lower"].tolist())
        fallback_location = next(
            (loc for loc in stage_locations if loc in stage_locations_with_results and loc != location),
            None
        )
        fallback_applied = any(loc != location for loc in stage_locations_with_results)
        emergency_filter_applied = severity == "high" and stage_name in {
            "expanded_strict_scope",
            "relaxed_specialty_emergency_only"
        }

        metadata = {
            "fallback_applied": fallback_applied,
            "fallback_location": fallback_location,
            "emergency_filter_applied": emergency_filter_applied,
            "specialty_requested": specialty,
            "strict_first_pass_applied": True,
            "strict_result_empty": True,
            "strict_message": strict_message,
            "guaranteed_fallback_applied": True,
            "guaranteed_fallback_stage": stage_name,
            "guaranteed_fallback_reason": stage_reason,
            "strict_emergency_unavailable": severity == "high"
        }

        fallback_response = _build_hospital_response_with_metadata(
            stage_df,
            requested_location=location,
            ordered_locations=stage_locations,
            metadata=metadata
        )
        if fallback_response.get("total_matches", 0) > 0:
            return fallback_response

    strict_results["metadata"] = {
        **strict_metadata,
        "strict_first_pass_applied": True,
        "strict_result_empty": True,
        "strict_message": strict_message,
        "guaranteed_fallback_applied": False,
        "guaranteed_fallback_stage": None,
        "strict_emergency_unavailable": severity == "high"
    }
    return strict_results


def _to_float_or_none(value):
    """
    Parse numeric values safely.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number == float("inf") or number == float("-inf"):
        return None
    return number


def _proximity_sort_key(item: dict):
    """
    Sort helper: distance ascending, then type/name for deterministic ordering.
    """
    distance = item.get("distance_km")
    return (
        distance is None,
        float("inf") if distance is None else float(distance),
        str(item.get("type", "")),
        str(item.get("name", "")),
    )


def _slug_token(value: str) -> str:
    text = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return token or "na"


def _build_unique_result_id(prefix: str, parts: list, used_ids: set) -> str:
    base = "-".join([_slug_token(prefix)] + [_slug_token(part) for part in parts if str(part).strip()])
    if not base:
        base = _slug_token(prefix)

    candidate = base
    counter = 2
    while candidate in used_ids:
        candidate = f"{base}-{counter}"
        counter += 1
    used_ids.add(candidate)
    return candidate


def _hospital_brand_key(hospital_name: str) -> str:
    """
    Build a stable brand key so chain variants (e.g., Sahyadri + area suffixes)
    can be deduplicated to one nearest result.
    """
    normalized = _normalize_specialty_text(hospital_name)
    if not normalized:
        return ""

    tokens = normalized.split()
    location_tokens = set()
    for location_name in LOCATION_COORDS.keys():
        location_tokens.update(_normalize_specialty_text(location_name).split())

    stop_tokens = {
        "hospital",
        "hospitals",
        "clinic",
        "medical",
        "center",
        "centre",
        "super",
        "speciality",
        "specialty",
        "multispeciality",
        "multispecialty",
        "and",
        "the",
        "of",
    }
    brand_alias_tokens = {
        "citi": "city",
        "sahyadree": "sahyadri",
    }

    brand_tokens = []
    for token in tokens:
        token = brand_alias_tokens.get(token, token)
        if token in stop_tokens:
            continue
        if token in location_tokens:
            continue
        if token.isdigit():
            continue
        brand_tokens.append(token)
        if len(brand_tokens) >= 3:
            break

    if not brand_tokens:
        brand_tokens = tokens[:2]

    return " ".join(brand_tokens).strip()


def _dedupe_hospital_records_by_brand(records: list, top_k: int = None) -> list:
    """
    Keep one nearest entry per hospital brand while preserving input order.
    Works for both keys:
      - recommendation payloads (`hospital_name`)
      - closest results (`name`)
    """
    if not records:
        return []

    deduped = []
    seen_brands = set()
    for record in records:
        hospital_name = str(
            record.get("hospital_name")
            or record.get("name")
            or ""
        ).strip()
        brand_key = _hospital_brand_key(hospital_name)
        if not brand_key:
            brand_key = _normalize_specialty_text(hospital_name)

        if brand_key and brand_key in seen_brands:
            continue
        if brand_key:
            seen_brands.add(brand_key)

        deduped.append(record)
        if top_k is not None and len(deduped) >= int(top_k):
            break

    return deduped


def build_closest_care_results(
    input_data: dict,
    origin_latitude=None,
    origin_longitude=None,
    top_k: int = DEFAULT_TOP_K,
) -> list:
    """
    Build up to `top_k` closest mixed doctor/hospital results for requested specialty.
    If specialty matches are fewer than top_k, backfill with closest general physicians.
    """
    specialty_key = canonicalize_specialty(input_data.get("specialty", ""))
    location_key = normalize_location_name(input_data.get("location", ""))
    if not specialty_key:
        return []

    ref_lat = _to_float_or_none(origin_latitude)
    ref_lon = _to_float_or_none(origin_longitude)
    if ref_lat is None or ref_lon is None:
        known_coords = LOCATION_COORDS.get(location_key)
        if known_coords:
            ref_lat, ref_lon = float(known_coords[0]), float(known_coords[1])

    def compute_distance(lat, lon):
        latitude = _to_float_or_none(lat)
        longitude = _to_float_or_none(lon)
        if latitude is None or longitude is None:
            return None
        if ref_lat is None or ref_lon is None:
            return None
        return _haversine_distance_km(ref_lat, ref_lon, latitude, longitude)

    specialty_candidates = []
    gp_fallback_candidates = []

    try:
        doctors_df = load_doctors_data().copy()
    except Exception:
        doctors_df = pd.DataFrame(columns=["name", "specialty", "location", "consultation_fee", "rating_score", "contact_number"])

    if not doctors_df.empty:
        doctors_df["specialty_canonical"] = doctors_df["specialty"].apply(canonicalize_specialty)

        for _, row in doctors_df.iterrows():
            doctor_specialty = row.get("specialty_canonical")
            location_value = str(row.get("location", "")).strip().lower()
            location_coords = LOCATION_COORDS.get(location_value)
            if not location_coords:
                continue

            latitude = float(location_coords[0])
            longitude = float(location_coords[1])
            distance_km = compute_distance(latitude, longitude)

            item = {
                "name": str(row.get("name", "Doctor")).strip() or "Doctor",
                "type": "doctor",
                "specialty": _display_specialty(doctor_specialty or row.get("specialty", "")),
                "distance_km": None if distance_km is None else round(distance_km, 2),
                "lat": latitude,
                "lng": longitude,
                "is_fallback": False,
                "location": location_value,
                "consultation_fee": int(row.get("consultation_fee", 0) or 0),
                "rating_score": _to_float_or_none(row.get("rating_score")),
                "contact_number": _get_valid_contact_number(row.get("contact_number")),
                "_id_parts": [row.get("name"), location_value, "doctor"],
            }

            if doctor_specialty == specialty_key:
                specialty_candidates.append(item)
            elif doctor_specialty == "general physician":
                fallback_item = {**item, "specialty": _display_specialty("general physician"), "is_fallback": True}
                gp_fallback_candidates.append(fallback_item)

    hospitals_file = _resolve_data_file("hospitals.csv")
    if hospitals_file.exists():
        try:
            hospitals_df = pd.read_csv(hospitals_file)
            hospitals_df = _prepare_hospitals_dataframe(hospitals_df)
        except Exception:
            hospitals_df = pd.DataFrame()
    else:
        hospitals_df = pd.DataFrame()

    if not hospitals_df.empty:
        for _, row in hospitals_df.iterrows():
            if not _hospital_matches_specialty(row.get("specialties_available", ""), specialty_key):
                continue

            latitude = _to_float_or_none(row.get("latitude"))
            longitude = _to_float_or_none(row.get("longitude"))
            if latitude is None or longitude is None:
                continue

            distance_km = compute_distance(latitude, longitude)
            specialty_candidates.append({
                "name": str(row.get("hospital_name", "Hospital")).strip() or "Hospital",
                "type": "hospital",
                "specialty": _display_specialty(specialty_key),
                "distance_km": None if distance_km is None else round(distance_km, 2),
                "lat": latitude,
                "lng": longitude,
                "is_fallback": False,
                "location": str(row.get("location", "")).strip(),
                "hospital_type": str(row.get("hospital_type", "")).strip(),
                "contact_number": _get_valid_contact_number(row.get("contact_number")),
                "_brand_key": _hospital_brand_key(row.get("hospital_name", "")),
                "_id_parts": [row.get("hospital_name"), row.get("location"), "hospital"],
            })

    specialty_candidates.sort(key=_proximity_sort_key)
    specialty_hospitals = [item for item in specialty_candidates if item.get("type") == "hospital"]
    deduped_hospitals = _dedupe_hospital_records_by_brand(specialty_hospitals)
    specialty_doctors = [item for item in specialty_candidates if item.get("type") != "hospital"]
    deduped_doctors = _dedupe_doctor_dicts(specialty_doctors, preferred_specialty=specialty_key)

    specialty_candidates = deduped_doctors + deduped_hospitals
    specialty_candidates.sort(key=_proximity_sort_key)
    selected = specialty_candidates[:top_k]

    if len(selected) < top_k and specialty_key != "general physician":
        gp_fallback_candidates.sort(key=_proximity_sort_key)
        gp_fallback_candidates = _dedupe_doctor_dicts(
            gp_fallback_candidates,
            preferred_specialty="general physician",
        )
        seen_fallback_names = {
            _normalize_doctor_name_key(item.get("name"))
            for item in selected
            if item.get("type") == "doctor"
        }
        for candidate in gp_fallback_candidates:
            name_key = _normalize_doctor_name_key(candidate.get("name"))
            if name_key and name_key in seen_fallback_names:
                continue
            selected.append(candidate)
            if name_key:
                seen_fallback_names.add(name_key)
            if len(selected) >= top_k:
                break

    selected.sort(key=_proximity_sort_key)
    selected = selected[:top_k]

    # Final pass: hospitals stay brand-deduped; doctors stay unique by name.
    final_doctors = _dedupe_doctor_dicts(
        [item for item in selected if item.get("type") == "doctor"],
        preferred_specialty=specialty_key,
    )
    final_hospitals = [item for item in selected if item.get("type") == "hospital"]
    selected = final_doctors + final_hospitals
    selected.sort(key=_proximity_sort_key)
    selected = selected[:top_k]

    used_ids = set()
    results = []
    for item in selected:
        item_id = _build_unique_result_id(item.get("type", "result"), item.get("_id_parts", []), used_ids)
        results.append({
            "id": item_id,
            "name": item.get("name"),
            "type": item.get("type"),
            "specialty": item.get("specialty"),
            "distance_km": item.get("distance_km"),
            "lat": item.get("lat"),
            "lng": item.get("lng"),
            "is_fallback": bool(item.get("is_fallback", False)),
            "location": item.get("location"),
            "contact_number": item.get("contact_number"),
            "consultation_fee": item.get("consultation_fee"),
            "rating_score": item.get("rating_score"),
            "hospital_type": item.get("hospital_type"),
        })

    return results


def _finalize_recommendation_doctor_lists(response: dict, preferred_specialty: str = "") -> dict:
    """
    Ensure doctor names are unique across recommended, optional, and closest lists.
    """
    if not isinstance(response, dict) or response.get("error"):
        return response

    preferred = canonicalize_specialty(preferred_specialty) or canonicalize_specialty(
        (response.get("metadata") or {}).get("query_specialty")
    )

    primary = _dedupe_doctor_dicts(response.get("recommended_doctors") or [], preferred)
    primary_names = {
        _normalize_doctor_name_key(doctor.get("name"))
        for doctor in primary
        if _normalize_doctor_name_key(doctor.get("name"))
    }

    optional_block = response.get("optional_nearby_doctors")
    optional_doctors = []
    if isinstance(optional_block, dict):
        optional_doctors = [
            doctor
            for doctor in _dedupe_doctor_dicts(
                optional_block.get("recommended_doctors") or [],
                preferred,
            )
            if _normalize_doctor_name_key(doctor.get("name")) not in primary_names
        ]

    claimed_names = primary_names | {
        _normalize_doctor_name_key(doctor.get("name"))
        for doctor in optional_doctors
        if _normalize_doctor_name_key(doctor.get("name"))
    }

    cleaned_closest = []
    seen_closest_names = set()
    for item in response.get("closest_results") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() == "doctor":
            name_key = _normalize_doctor_name_key(item.get("name"))
            if not name_key or name_key in claimed_names or name_key in seen_closest_names:
                continue
            seen_closest_names.add(name_key)
        cleaned_closest.append(item)

    if "recommended_doctors" in response:
        response["recommended_doctors"] = primary
    if isinstance(optional_block, dict):
        response["optional_nearby_doctors"] = {
            **optional_block,
            "recommended_doctors": optional_doctors,
            "returned_count": len(optional_doctors),
            "total_doctors_available": optional_block.get(
                "total_doctors_available",
                len(optional_doctors),
            ),
        }
    if "closest_results" in response:
        response["closest_results"] = cleaned_closest
    return response


def generate_recommendation_response(input_data: dict):
    """
    Generate a comprehensive recommendation response with severity-based routing.
    
    Routes requests based on severity level:
    - "high": Hospital-only with strict emergency filtering first, then guaranteed fallback
    - "medium": Hospitals always included (strict first, guaranteed fallback) + optional/fallback doctors
    - "low": Doctor recommendations only for routine checkups
    
    Args:
        input_data (dict): Dictionary containing:
            - "specialty" (str): Desired medical specialty (required)
            - "location" (str OR dict): Location as string name OR dict with "latitude" and "longitude" (required)
            - "severity" (str): Severity level - "low", "medium", or "high" (optional, default: "medium")
    
    Returns:
        dict: Dictionary with either recommended_doctors or recommended_hospitals
              based on severity-based routing logic, or error dict
    """
    # INPUT VALIDATION
    if not isinstance(input_data, dict):
        return {"error": "Invalid input format."}
    
    specialty = canonicalize_specialty(input_data.get("specialty", ""))
    if not specialty:
        return {"error": "Specialty is required."}
    
    # LOCATION HANDLING - Support both string and lat/lng coordinates
    location_input = input_data.get("location")
    if not location_input:
        return {"error": "Location is required."}
    
    coordinates_out_of_coverage = False
    query_latitude = None
    query_longitude = None

    # Check if location is provided as coordinates
    if isinstance(location_input, dict):
        latitude = location_input.get("latitude")
        longitude = location_input.get("longitude")
        
        if latitude is None or longitude is None:
            return {"error": "Location coordinates must include both latitude and longitude."}
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return {"error": "Invalid latitude or longitude values."}
        query_latitude = latitude
        query_longitude = longitude
        
        # Convert coordinates to nearest known location name.
        detected_location = convert_coordinates_to_location(latitude, longitude)
        if detected_location is None:
            # Out-of-coverage coordinates: use nearest supported region so
            # recommendation flows still return actionable hospitals/doctors.
            detected_location = convert_coordinates_to_location(
                latitude,
                longitude,
                max_distance_km=20000.0,
            )
            if detected_location is None:
                return {
                    "error": "No known locations found near the provided coordinates. Please try a different location.",
                    "metadata": {
                        "provided_coordinates": {"latitude": latitude, "longitude": longitude}
                    }
                }
            coordinates_out_of_coverage = True
            location_source = "coordinates_nearest_fallback"
        else:
            location_source = "coordinates"
        location = detected_location
    else:
        # Location provided as string
        location = normalize_location_name(location_input)
        if not location:
            return {"error": "Location is required."}
        location_source = "name"
        location_coords = LOCATION_COORDS.get(location)
        if location_coords:
            query_latitude = float(location_coords[0])
            query_longitude = float(location_coords[1])
    
    # SEVERITY VALIDATION
    severity = input_data.get("severity", "medium").strip().lower()
    if severity not in ["low", "medium", "high"]:
        return {"error": "Invalid severity. Must be low, medium, or high."}
    
    # Prepare cleaned input for downstream functions
    cleaned_input = {
        "specialty": specialty,
        "location": location,
        "severity": severity
    }
    closest_results = build_closest_care_results(
        cleaned_input,
        origin_latitude=query_latitude,
        origin_longitude=query_longitude,
        top_k=DEFAULT_TOP_K,
    )
    
    # SEVERITY-BASED ROUTING
    if severity == "high":
        # HIGH: Strict logic first, then staged guaranteed fallback if strict is empty.
        hospital_results = recommend_hospitals_with_guaranteed_fallback(cleaned_input)
        hospital_count = hospital_results.get("total_matches", 0)
        hospital_metadata = hospital_results.get("metadata", {}) or {}
        
        response = {
            "care_setting": "hospital",
            "metadata": {
                "query_specialty": specialty,
                "query_location": location,
                "query_severity": severity,
                "severity": severity,
                "recommendation_type": "hospital_only",
                "emergency_filter_applied": hospital_metadata.get("emergency_filter_applied", True),
                "strict_emergency_unavailable": hospital_metadata.get(
                    "strict_emergency_unavailable",
                    hospital_metadata.get("strict_result_empty", False)
                ),
                "strict_result_empty": hospital_metadata.get("strict_result_empty", False),
                "guaranteed_fallback_applied": hospital_metadata.get("guaranteed_fallback_applied", False),
                "guaranteed_fallback_stage": hospital_metadata.get("guaranteed_fallback_stage"),
                "location_source": location_source,
                "detected_location": location if location_source.startswith("coordinates") else None,
                "coordinates_out_of_coverage": coordinates_out_of_coverage
            },
            "recommended_hospitals": hospital_results.get("recommended_hospitals", []),
            "total_hospitals_available": hospital_count,
            "closest_results": closest_results
        }
        if "message" in hospital_results:
            response["message"] = hospital_results["message"]
        return _finalize_recommendation_doctor_lists(response, preferred_specialty=specialty)
    
    elif severity == "medium":
        # MEDIUM: Strict logic first, then staged guaranteed fallback.
        hospital_results = recommend_hospitals_with_guaranteed_fallback(cleaned_input)
        hospital_count = hospital_results.get("total_matches", 0)
        hospital_metadata = hospital_results.get("metadata", {}) or {}
        doctor_recommendations = recommend_doctors(cleaned_input)
        recommended_doctors = doctor_recommendations.get("recommended_doctors", [])
        doctor_fallback_metadata = doctor_recommendations.get("metadata", {})
        doctor_fallback_type = doctor_fallback_metadata.get("fallback_type")
        
        if hospital_count > 0:
            # Hospitals found - return hospital recommendations plus optional nearby doctors
            return _finalize_recommendation_doctor_lists(
                {
                    "care_setting": "hospital",
                    "metadata": {
                        "query_specialty": specialty,
                        "query_location": location,
                        "query_severity": severity,
                        "severity": severity,
                        "recommendation_type": "hospital_primary",
                        "optional_doctors_available": len(recommended_doctors) > 0,
                        "strict_result_empty": hospital_metadata.get("strict_result_empty", False),
                        "guaranteed_fallback_applied": hospital_metadata.get("guaranteed_fallback_applied", False),
                        "guaranteed_fallback_stage": hospital_metadata.get("guaranteed_fallback_stage"),
                        "location_source": location_source,
                        "detected_location": location if location_source.startswith("coordinates") else None,
                        "coordinates_out_of_coverage": coordinates_out_of_coverage
                    },
                    "recommended_hospitals": hospital_results.get("recommended_hospitals", []),
                    "total_hospitals_available": hospital_count,
                    "closest_results": closest_results,
                    "optional_nearby_doctors": {
                        "recommended_doctors": recommended_doctors,
                        "total_doctors_available": doctor_recommendations.get("total_matches", 0),
                        "returned_count": len(recommended_doctors),
                        "fallback_applied": doctor_fallback_metadata.get("fallback_applied", False),
                        "fallback_location": doctor_fallback_metadata.get("fallback_location"),
                        "fallback_type": doctor_fallback_type,
                        "cost_summary": _generate_cost_summary_from_recommendations(recommended_doctors)
                    }
                },
                preferred_specialty=specialty,
            )
        else:
            # No hospitals found - fallback to doctors
            cost_summary = _generate_cost_summary_from_recommendations(recommended_doctors)
            
            returned_count = len(recommended_doctors)
            total_matches = doctor_recommendations.get("total_matches", 0)
            fallback_type = doctor_fallback_metadata.get("fallback_type")
            
            metadata = {
                "query_specialty": specialty,
                "query_location": location,
                "query_severity": severity,
                "severity": severity,
                "recommendation_type": "doctor_fallback",
                "total_doctors_available": total_matches,
                "returned_count": returned_count,
                "fallback_applied": doctor_fallback_metadata.get("fallback_applied", False),
                "fallback_location": doctor_fallback_metadata.get("fallback_location"),
                "fallback_type": fallback_type,
                "strict_result_empty": hospital_metadata.get("strict_result_empty", True),
                "guaranteed_fallback_applied": hospital_metadata.get("guaranteed_fallback_applied", False),
                "guaranteed_fallback_stage": hospital_metadata.get("guaranteed_fallback_stage"),
                "location_source": location_source,
                "detected_location": location if location_source.startswith("coordinates") else None,
                "coordinates_out_of_coverage": coordinates_out_of_coverage
            }
            
            if fallback_type == "general_physician":
                metadata["original_specialty"] = doctor_fallback_metadata.get("original_specialty", specialty)
            
            response = {
                "care_setting": "clinic",
                "metadata": metadata,
                "recommended_hospitals": hospital_results.get("recommended_hospitals", []),
                "total_hospitals_available": hospital_count,
                "recommended_doctors": recommended_doctors,
                "cost_summary": cost_summary,
                "closest_results": closest_results
            }
            
            if "message" in hospital_results:
                response["hospital_message"] = hospital_results["message"]
            if "message" in doctor_recommendations:
                response["message"] = doctor_recommendations["message"]
            
            return _finalize_recommendation_doctor_lists(response, preferred_specialty=specialty)
    
    else:  # severity == "low"
        # LOW: Doctor engine only, with hospital fallback if no doctors found
        doctor_recommendations = recommend_doctors(cleaned_input)
        doctor_count = len(doctor_recommendations.get("recommended_doctors", []))
        
        # If no doctors found, fallback to hospitals
        if doctor_count == 0:
            hospital_results = recommend_hospitals(cleaned_input)
            hospital_count = hospital_results.get("total_matches", 0)
            
            if hospital_count > 0:
                # Return hospital recommendations as fallback
                return _finalize_recommendation_doctor_lists(
                    {
                        "care_setting": "hospital",
                        "metadata": {
                            "query_specialty": specialty,
                            "query_location": location,
                            "query_severity": severity,
                            "severity": severity,
                            "recommendation_type": "hospital_fallback",
                            "reason": "No doctors available, suggesting hospitals instead",
                            "location_source": location_source,
                            "detected_location": location if location_source.startswith("coordinates") else None,
                            "coordinates_out_of_coverage": coordinates_out_of_coverage
                        },
                        "recommended_hospitals": hospital_results.get("recommended_hospitals", []),
                        "total_hospitals_available": hospital_count,
                        "closest_results": closest_results
                    },
                    preferred_specialty=specialty,
                )
        
        # Otherwise, continue with doctor recommendations
        recommended_doctors = doctor_recommendations.get("recommended_doctors", [])
        cost_summary = _generate_cost_summary_from_recommendations(recommended_doctors)
        
        returned_count = len(recommended_doctors)
        total_matches = doctor_recommendations.get("total_matches", 0)
        fallback_metadata = doctor_recommendations.get("metadata", {})
        fallback_type = fallback_metadata.get("fallback_type")
        
        metadata = {
            "query_specialty": specialty,
            "query_location": location,
            "query_severity": severity,
            "severity": severity,
            "recommendation_type": "doctor_only",
            "total_doctors_available": total_matches,
            "returned_count": returned_count,
            "fallback_applied": fallback_metadata.get("fallback_applied", False),
            "fallback_location": fallback_metadata.get("fallback_location"),
            "fallback_type": fallback_type,
            "location_source": location_source,
            "detected_location": location if location_source.startswith("coordinates") else None,
            "coordinates_out_of_coverage": coordinates_out_of_coverage
        }
        
        if fallback_type == "general_physician" or fallback_type == "general_physician_nearby":
            metadata["original_specialty"] = fallback_metadata.get("original_specialty", specialty)
        
        response = {
            "care_setting": "clinic",
            "metadata": metadata,
            "recommended_doctors": recommended_doctors,
            "cost_summary": cost_summary,
            "closest_results": closest_results
        }
        
        if "message" in doctor_recommendations:
            response["message"] = doctor_recommendations["message"]
        
        return _finalize_recommendation_doctor_lists(response, preferred_specialty=specialty)
