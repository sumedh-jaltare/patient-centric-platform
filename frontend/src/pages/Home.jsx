import { useRef, useState } from 'react';
import SymptomForm from '../components/SymptomForm';
import KnownCareForm from '../components/KnownCareForm';
import CarePathSelector from '../components/CarePathSelector';
import LoadingScreen from '../components/LoadingScreen';
import ResultDashboard from '../components/ResultDashboard';
import { analyzeSymptoms } from '../services/api';
import { calculateDistance } from '../services/geolocation';
import '../styles/Home.css';

const COORDINATES_PATTERN = /\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?/;

const toNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeDoctorRatingToTen = (value) => {
  const numeric = toNumber(value);
  if (numeric == null) return 8.0;

  if (numeric <= 1) return Math.max(0, Math.min(10, numeric * 10));
  if (numeric <= 5) return Math.max(0, Math.min(10, numeric * 2));
  if (numeric <= 10) return Math.max(0, Math.min(10, numeric));
  if (numeric <= 100) return Math.max(0, Math.min(10, numeric / 10));
  return 8.0;
};

const normalizeUrgency = (value) => {
  const urgency = String(value || 'low').toLowerCase().trim();
  if (urgency === 'high' || urgency === 'medium' || urgency === 'low') {
    return urgency;
  }
  return 'low';
};

const mapHospitalCostLevel = (hospitalType, averageFee) => {
  const type = String(hospitalType || '').toLowerCase();
  if (type.includes('private') || type.includes('corporate')) return 'high';
  if (type.includes('public') || type.includes('government')) return 'low';

  if (averageFee == null) return 'medium';
  if (averageFee <= 500) return 'low';
  if (averageFee <= 1000) return 'medium';
  return 'high';
};

const parseOptionalLocation = (locationText) => {
  const input = String(locationText || '').trim();
  if (!input) return null;

  const match = input.match(COORDINATES_PATTERN);
  if (!match) return null;

  const latitude = toNumber(match[1]);
  const longitude = toNumber(match[2]);

  if (latitude == null || longitude == null) return null;
  return { latitude, longitude };
};

const parseLocationFromMode = (locationMode, locationValue) => {
  const input = String(locationValue || '').trim();
  if (!input) return null;

  if (locationMode === 'coordinates') {
    return parseOptionalLocation(input);
  }

  return { city: input };
};

const normalizeLocationText = (value) => String(value || '').trim().replace(/,\s*$/, '');

const extractDoctorLocation = (doctor) => {
  const candidates = [
    doctor?.location,
    doctor?.doctors_location,
    doctor?.facility_location,
    doctor?.floor,
    doctor?.hospital_location,
  ];

  for (const candidate of candidates) {
    const normalized = normalizeLocationText(candidate);
    if (normalized) return normalized;
  }

  return 'N/A';
};

const KNOWN_INPUT_LABELS = {
  diagnosis: 'Diagnosis',
  specialist: 'Specialist',
  procedure: 'Procedure / Surgery',
};

const HOSPITAL_BRAND_STOPWORDS = new Set([
  'hospital',
  'hospitals',
  'clinic',
  'medical',
  'center',
  'centre',
  'super',
  'speciality',
  'specialty',
  'multispeciality',
  'multispecialty',
  'and',
  'the',
  'of',
]);

const HOSPITAL_BRAND_ALIASES = {
  citi: 'city',
  sahyadree: 'sahyadri',
};

const tokenizeText = (value) =>
  String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => HOSPITAL_BRAND_ALIASES[token] || token);

const buildHospitalBrandKey = (name, location) => {
  const tokens = tokenizeText(name);
  if (tokens.length === 0) return '';

  const locationTokens = new Set(tokenizeText(location));
  const brandTokens = [];
  tokens.forEach((token) => {
    if (HOSPITAL_BRAND_STOPWORDS.has(token)) return;
    if (locationTokens.has(token)) return;
    if (/^\d+$/.test(token)) return;
    if (brandTokens.length < 3) brandTokens.push(token);
  });

  if (brandTokens.length === 0) {
    return tokens.slice(0, 2).join(' ').trim();
  }
  return brandTokens.join(' ').trim();
};

const dedupeHospitalsByBrand = (hospitals) => {
  if (!Array.isArray(hospitals) || hospitals.length === 0) return [];

  const seen = new Set();
  const deduped = [];
  hospitals.forEach((hospital) => {
    const key = buildHospitalBrandKey(hospital?.name || hospital?.hospital_name, hospital?.location);
    const dedupeKey = key || `${String(hospital?.name || hospital?.hospital_name || '').toLowerCase()}|${String(hospital?.location || '').toLowerCase()}`;
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    deduped.push(hospital);
  });

  return deduped;
};

const formatCostEstimate = (scrapingOutput) => {
  const costSummary =
    scrapingOutput?.cost_summary || scrapingOutput?.optional_nearby_doctors?.cost_summary;

  if (!costSummary) return 'Contact provider for pricing';

  const minFee = toNumber(costSummary.min_fee);
  const maxFee = toNumber(costSummary.max_fee);
  const averageFee = toNumber(costSummary.average_fee);
  const costBand = costSummary.cost_band;

  if (minFee != null && maxFee != null) {
    const base = `Rs.${Math.round(minFee)} - Rs.${Math.round(maxFee)}`;
    return costBand ? `${base} (${costBand})` : base;
  }
  if (averageFee != null) {
    return `Approx. Rs.${Math.round(averageFee)}${costBand ? ` (${costBand})` : ''}`;
  }
  return 'Contact provider for pricing';
};

const normalizeDoctorName = (value) =>
  String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');

const normalizePhoneDigits = (value) => String(value || '').replace(/\D/g, '');

const buildDoctorDedupeKey = (doctor) => {
  const name = normalizeDoctorName(doctor?.name);
  const location = extractDoctorLocation(doctor).toLowerCase();
  const phone = normalizePhoneDigits(doctor?.phone || doctor?.contact_number);
  // Same person often appears once from recommendations and once from closest_results
  // with different specialty casing / phone formatting.
  return `${name}|${location}|${phone}`;
};

const doctorRankScore = (doctor, preferredSpecialty = '') => {
  const rating = toNumber(doctor?.rating) ?? -1;
  const distance = toNumber(doctor?.distance_km);
  const distanceScore = distance == null ? 0 : Math.max(0, 100 - distance);
  const fallbackPenalty = doctor?.is_fallback || String(doctor?.availability || '').includes('Fallback') ? 0 : 10;
  const phoneBonus = normalizePhoneDigits(doctor?.phone || doctor?.contact_number) ? 1 : 0;
  const specialty = String(doctor?.specialty || '').trim().toLowerCase();
  const preferred = String(preferredSpecialty || '').trim().toLowerCase();
  const preferredBonus = preferred && specialty === preferred ? 8 : 0;
  // Prefer the primary care label when the same doctor is listed under both GP and IM.
  const specialtyBonus = specialty === 'general physician' ? 5 : specialty === 'internal medicine' ? 2 : 0;
  return rating * 10 + distanceScore + fallbackPenalty + phoneBonus + preferredBonus + specialtyBonus;
};

const formatSpecialtyLabel = (value) => {
  const text = String(value || 'General Physician').trim();
  if (!text) return 'General Physician';
  return text
    .toLowerCase()
    .split(/\s+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
};

const mapDoctors = (rawDoctors, userLocation) => {
  if (!Array.isArray(rawDoctors)) return [];

  const mapped = [];

  rawDoctors.forEach((doctor) => {
    const doctorLocation = extractDoctorLocation(doctor);

    const fee = toNumber(doctor?.consultation_fee);
    const latitude = toNumber(doctor?.latitude);
    const longitude = toNumber(doctor?.longitude);

    let distanceKm = toNumber(doctor?.distance_km);
    if (
      distanceKm == null &&
      latitude != null &&
      longitude != null &&
      userLocation?.latitude != null &&
      userLocation?.longitude != null
    ) {
      distanceKm = calculateDistance(
        userLocation.latitude,
        userLocation.longitude,
        latitude,
        longitude
      );
    }

    const ratingOutOfTen = normalizeDoctorRatingToTen(doctor?.rating_score ?? doctor?.rating);

    mapped.push({
      name: doctor?.name || 'Doctor',
      specialty: formatSpecialtyLabel(doctor?.specialty || 'General Physician'),
      experience_years: doctor?.experience_years ?? null,
      availability: 'Recommended',
      rating: Number(ratingOutOfTen.toFixed(1)),
      cost: fee == null ? 'On request' : `Rs.${Math.round(fee)}`,
      phone: doctor?.contact_number || 'Not available',
      location: doctorLocation,
      consultation_fee: fee,
      distance_km: distanceKm == null ? null : Number(distanceKm.toFixed(1)),
      latitude,
      longitude,
    });
  });

  return dedupeDoctors(mapped);
};

const dedupeDoctors = (doctors, preferredSpecialty = '') => {
  if (!Array.isArray(doctors) || doctors.length === 0) return [];

  const bestByKey = new Map();
  doctors.forEach((doctor) => {
    if (!normalizeDoctorName(doctor?.name)) return;

    const key = buildDoctorDedupeKey(doctor);
    const existing = bestByKey.get(key);
    if (!existing || doctorRankScore(doctor, preferredSpecialty) > doctorRankScore(existing, preferredSpecialty)) {
      bestByKey.set(key, doctor);
    }
  });

  // Collapse remaining same-name repeats across slightly different location strings.
  const bestByName = new Map();
  bestByKey.forEach((doctor) => {
    const nameKey = normalizeDoctorName(doctor?.name);
    const existing = bestByName.get(nameKey);
    if (!existing || doctorRankScore(doctor, preferredSpecialty) > doctorRankScore(existing, preferredSpecialty)) {
      bestByName.set(nameKey, doctor);
    }
  });

  return Array.from(bestByName.values());
};

const buildHospitalsFromRecommendations = (rawHospitals, mappedDoctors, userLocation) => {
  if (!Array.isArray(rawHospitals) || rawHospitals.length === 0) return [];

  const doctorsByLocation = mappedDoctors.reduce((acc, doctor) => {
    const key = String(doctor.location || '').toLowerCase().trim();
    if (!key) return acc;
    if (!acc[key]) acc[key] = [];
    acc[key].push(doctor);
    return acc;
  }, {});

  return rawHospitals.map((hospital, index) => {
    const latitude = toNumber(hospital?.latitude);
    const longitude = toNumber(hospital?.longitude);
    const hasCoordinates = latitude != null && longitude != null;
    const locationKey = String(hospital?.location || '').toLowerCase().trim();
    const locationDoctors = doctorsByLocation[locationKey] || [];
    const fallbackDoctors = index === 0 ? mappedDoctors.slice(0, 3) : [];
    const linkedDoctors = (locationDoctors.length > 0 ? locationDoctors : fallbackDoctors).slice(0, 3);

    let distanceKm = toNumber(hospital?.distance_km);
    if (
      distanceKm == null &&
      hasCoordinates &&
      userLocation?.latitude != null &&
      userLocation?.longitude != null
    ) {
      distanceKm = calculateDistance(
        userLocation.latitude,
        userLocation.longitude,
        latitude,
        longitude
      );
    }

    const score = toNumber(hospital?.score);
    const rating =
      score == null
        ? 4.2
        : Math.max(3.5, Math.min(5, Number((3 + score * 2).toFixed(1))));

    const averageDoctorFee =
      linkedDoctors.length > 0
        ? (() => {
            const fees = linkedDoctors
              .map((doctor) => doctor.consultation_fee)
              .filter((fee) => fee != null);
            if (fees.length === 0) return null;
            return fees.reduce((sum, fee) => sum + fee, 0) / fees.length;
          })()
        : null;

    return {
      name: hospital?.hospital_name || 'Hospital',
      location: hospital?.location || 'N/A',
      distance_km: distanceKm == null ? 'N/A' : Number(distanceKm.toFixed(1)),
      rating,
      cost_level: mapHospitalCostLevel(hospital?.hospital_type, averageDoctorFee),
      insurance_supported: true,
      coordinates: hasCoordinates ? [latitude, longitude] : null,
      doctors: linkedDoctors,
      hospital_type: hospital?.hospital_type || 'Unknown',
    };
  });
};

const buildHospitalsFromDoctors = (mappedDoctors, userLocation) => {
  if (!Array.isArray(mappedDoctors) || mappedDoctors.length === 0) return [];

  const groupedByLocation = mappedDoctors.reduce((acc, doctor) => {
    const key = String(doctor.location || 'Local Area').trim();
    if (!acc[key]) acc[key] = [];
    acc[key].push(doctor);
    return acc;
  }, {});

  return Object.entries(groupedByLocation).map(([location, doctors]) => {
    const firstWithCoords = doctors.find(
      (doctor) => doctor.latitude != null && doctor.longitude != null
    );

    let distanceKm = null;
    if (
      firstWithCoords &&
      userLocation?.latitude != null &&
      userLocation?.longitude != null
    ) {
      distanceKm = calculateDistance(
        userLocation.latitude,
        userLocation.longitude,
        firstWithCoords.latitude,
        firstWithCoords.longitude
      );
    }

    const rating =
      doctors.length > 0
        ? Number(
            (
              doctors.reduce((sum, doctor) => sum + (toNumber(doctor.rating) || 8), 0) /
              doctors.length /
              2
            ).toFixed(1)
          )
        : 4.0;

    const avgFee =
      doctors.length > 0
        ? (() => {
            const fees = doctors
              .map((doctor) => doctor.consultation_fee)
              .filter((fee) => fee != null);
            if (fees.length === 0) return null;
            return fees.reduce((sum, fee) => sum + fee, 0) / fees.length;
          })()
        : null;

    return {
      name: `Clinics near ${location}`,
      location,
      distance_km: distanceKm == null ? 'N/A' : Number(distanceKm.toFixed(1)),
      rating: Math.max(3.5, Math.min(5, rating)),
      cost_level: mapHospitalCostLevel('', avgFee),
      insurance_supported: true,
      coordinates:
        firstWithCoords && firstWithCoords.latitude != null && firstWithCoords.longitude != null
          ? [firstWithCoords.latitude, firstWithCoords.longitude]
          : null,
      doctors: doctors.slice(0, 3),
      hospital_type: 'Clinic Network',
    };
  });
};

const mapClosestResults = (rawResults, userLocation) => {
  if (!Array.isArray(rawResults) || rawResults.length === 0) {
    return { doctors: [], hospitals: [] };
  }

  const doctors = [];
  const hospitals = [];
  const seenDoctorIds = new Set();
  const seenHospitalIds = new Set();
  const seenHospitalBrandKeys = new Set();

  rawResults.forEach((item) => {
    const id = String(item?.id || '').trim();
    const type = String(item?.type || '').toLowerCase().trim();
    const latitude = toNumber(item?.lat);
    const longitude = toNumber(item?.lng);

    let distanceKm = toNumber(item?.distance_km);
    if (
      distanceKm == null &&
      latitude != null &&
      longitude != null &&
      userLocation?.latitude != null &&
      userLocation?.longitude != null
    ) {
      distanceKm = calculateDistance(
        userLocation.latitude,
        userLocation.longitude,
        latitude,
        longitude
      );
    }

    if (type === 'doctor') {
      const doctorId = id || `${String(item?.name || 'doctor')}-${String(item?.location || '')}`;
      if (seenDoctorIds.has(doctorId)) return;
      seenDoctorIds.add(doctorId);

      const fee = toNumber(item?.consultation_fee);
      const ratingOutOfTen = normalizeDoctorRatingToTen(item?.rating_score);
      doctors.push({
        id: doctorId,
        name: item?.name || 'Doctor',
        specialty: formatSpecialtyLabel(item?.specialty || 'General Physician'),
        experience_years: null,
        availability: item?.is_fallback ? 'Fallback Recommendation' : 'Recommended',
        rating: Number((ratingOutOfTen ?? 8).toFixed(1)),
        cost: fee == null ? 'On request' : `Rs.${Math.round(fee)}`,
        phone: item?.contact_number || 'Not available',
        location: item?.location || 'N/A',
        consultation_fee: fee,
        distance_km: distanceKm == null ? null : Number(distanceKm.toFixed(1)),
        latitude,
        longitude,
        is_fallback: Boolean(item?.is_fallback),
      });
      return;
    }

    if (type === 'hospital') {
      const hospitalId = id || `${String(item?.name || 'hospital')}-${String(item?.location || '')}`;
      if (seenHospitalIds.has(hospitalId)) return;
      const brandKey = buildHospitalBrandKey(item?.name, item?.location);
      if (brandKey && seenHospitalBrandKeys.has(brandKey)) return;
      seenHospitalIds.add(hospitalId);
      if (brandKey) seenHospitalBrandKeys.add(brandKey);

      hospitals.push({
        id: hospitalId,
        name: item?.name || 'Hospital',
        location: item?.location || 'N/A',
        distance_km: distanceKm == null ? 'N/A' : Number(distanceKm.toFixed(1)),
        rating: 4.2,
        cost_level: mapHospitalCostLevel(item?.hospital_type, null),
        insurance_supported: true,
        coordinates: latitude != null && longitude != null ? [latitude, longitude] : null,
        doctors: [],
        hospital_type: item?.hospital_type || 'Unknown',
        specialty: item?.specialty || '',
        is_fallback: Boolean(item?.is_fallback),
      });
    }
  });

  doctors.sort((a, b) => {
    if (a.distance_km == null && b.distance_km == null) return 0;
    if (a.distance_km == null) return 1;
    if (b.distance_km == null) return -1;
    return a.distance_km - b.distance_km;
  });

  hospitals.sort((a, b) => {
    const aDistance = typeof a.distance_km === 'number' ? a.distance_km : null;
    const bDistance = typeof b.distance_km === 'number' ? b.distance_km : null;
    if (aDistance == null && bDistance == null) return 0;
    if (aDistance == null) return 1;
    if (bDistance == null) return -1;
    return aDistance - bDistance;
  });

  return { doctors, hospitals };
};

const normalizeAssessmentResponse = (response) => {
  const scrapingOutput =
    response?.scraping_recommendations && typeof response.scraping_recommendations === 'object'
      ? response.scraping_recommendations
      : null;

  const closestResults = Array.isArray(scrapingOutput?.closest_results)
    ? scrapingOutput.closest_results
    : [];
  const mappedFromClosest = mapClosestResults(closestResults, response?.user_location);

  const primaryDoctors = Array.isArray(scrapingOutput?.recommended_doctors)
    ? scrapingOutput.recommended_doctors
    : [];
  const optionalDoctors = Array.isArray(scrapingOutput?.optional_nearby_doctors?.recommended_doctors)
    ? scrapingOutput.optional_nearby_doctors.recommended_doctors
    : [];
  const preferredSpecialty = formatSpecialtyLabel(
    response?.recommended_specialty || 'General Physician'
  );
  const mappedDoctorsFromRecommendations = mapDoctors(
    [...primaryDoctors, ...optionalDoctors],
    response?.user_location
  );
  const mappedDoctors = dedupeDoctors(
    [...mappedDoctorsFromRecommendations, ...mappedFromClosest.doctors],
    preferredSpecialty
  );

  const hospitalsFromRecommendations = buildHospitalsFromRecommendations(
    scrapingOutput?.recommended_hospitals || [],
    mappedDoctors,
    response?.user_location
  );
  const hospitalsFromClosest = dedupeHospitalsByBrand(mappedFromClosest.hospitals);
  const dedupedRecommendations = dedupeHospitalsByBrand(hospitalsFromRecommendations);
  const fallbackHospitals =
    hospitalsFromClosest.length > 0
      ? hospitalsFromClosest
      : dedupedRecommendations.length > 0
      ? dedupedRecommendations
      : buildHospitalsFromDoctors(mappedDoctors, response?.user_location);

  return {
    urgency: normalizeUrgency(response?.urgency_level),
    emergency_flag: Boolean(response?.emergency_flag),
    recommendation: response?.reasoning || 'No recommendation available.',
    specialty: response?.recommended_specialty || 'General Physician',
    care_setting: response?.care_setting || 'clinic',
    symptoms_identified: Array.isArray(response?.symptoms_identified)
      ? response.symptoms_identified
      : [],
    reasoning: response?.reasoning || '',
    safety_advice: response?.safety_advice || '',
    disclaimer: response?.disclaimer || '',
    session_id: response?.session_id || '',
    user_location: response?.user_location || null,
    scraping_input: response?.scraping_input || null,
    scraping_recommendations: scrapingOutput,
    closest_results: closestResults,
    hospitals: fallbackHospitals,
    doctors: mappedDoctors,
    cost_estimate: formatCostEstimate(scrapingOutput),
  };
};

export default function Home() {
  const sessionIdRef = useRef(
    `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
  );
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState('entry');
  const [result, setResult] = useState(null);
  const [inputSummary, setInputSummary] = useState(null);
  const [inputLabel, setInputLabel] = useState('Your Symptoms');
  const [error, setError] = useState('');

  const handleSymptomSubmit = async ({ symptoms: userSymptoms, locationMode, locationValue }) => {
    setLoading(true);
    setError('');
    setInputLabel('Your Symptoms');
    setInputSummary(userSymptoms);

    try {
      const payload = {
        message: userSymptoms,
        session_id: sessionIdRef.current,
      };

      const parsedLocation = parseLocationFromMode(locationMode, locationValue);
      if (parsedLocation) {
        payload.location = parsedLocation;
      }

      const response = await analyzeSymptoms(payload);
      setResult(normalizeAssessmentResponse(response));
    } catch (apiError) {
      const message = String(apiError?.message || 'Unable to assess symptoms right now.');
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleKnownInputSubmit = async ({
    inputType,
    knownInput,
    details,
    locationMode,
    locationValue,
  }) => {
    setLoading(true);
    setError('');
    setInputLabel('Your Medical Input');
    setInputSummary(
      `${KNOWN_INPUT_LABELS[inputType] || 'Known Input'}: ${knownInput}${details ? `\nNotes: ${details}` : ''}`
    );

    try {
      const payload = {
        message: `Known ${inputType}: ${knownInput}${details ? `. Additional context: ${details}` : ''}.`,
        session_id: sessionIdRef.current,
      };

      const parsedLocation = parseLocationFromMode(locationMode, locationValue);
      if (parsedLocation) {
        payload.location = parsedLocation;
      }

      const response = await analyzeSymptoms(payload);
      setResult(normalizeAssessmentResponse(response));
    } catch (apiError) {
      const message = String(apiError?.message || 'Unable to assess this input right now.');
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPath = (path) => {
    setActiveView(path);
    setError('');
  };

  const handleBackClick = () => {
    setResult(null);
    setInputSummary(null);
    setInputLabel('Your Symptoms');
    setError('');
    setActiveView('entry');
  };

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <div className="home">
      {result ? (
        <ResultDashboard
          result={result}
          symptoms={inputSummary}
          inputLabel={inputLabel}
          onBackClick={handleBackClick}
        />
      ) : (
        <>
          {activeView === 'entry' && <CarePathSelector onSelectPath={handleSelectPath} />}

          {activeView === 'guided' && (
            <>
              {error && <div className="home-error">{error}</div>}
              <SymptomForm onSubmit={handleSymptomSubmit} onBack={handleBackClick} />
            </>
          )}

          {activeView === 'known' && (
            <>
              {error && <div className="home-error">{error}</div>}
              <KnownCareForm onSubmit={handleKnownInputSubmit} onBack={handleBackClick} />
            </>
          )}
        </>
      )}
    </div>
  );
}
