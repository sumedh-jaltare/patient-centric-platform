import { useMemo, useState } from 'react';
import '../styles/DoctorList.css';
import ReviewSummary from './ReviewSummary';

const toNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeLocationText = (value) => String(value || '').trim().replace(/,\s*$/, '');

const getDoctorLocationText = (doctor) => {
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

function normalizeDoctorRatingToTen(ratingValue) {
  const numeric =
    typeof ratingValue === 'string'
      ? Number(ratingValue.replace('%', '').trim())
      : Number(ratingValue);

  if (!Number.isFinite(numeric)) return null;

  if (numeric <= 1) return Math.max(0, Math.min(10, numeric * 10));
  if (numeric <= 5) return Math.max(0, Math.min(10, numeric * 2));
  if (numeric <= 10) return Math.max(0, Math.min(10, numeric));
  if (numeric <= 100) return Math.max(0, Math.min(10, numeric / 10));
  return null;
}

function formatDoctorRatingOutOfTen(ratingValue) {
  const normalized = normalizeDoctorRatingToTen(ratingValue);
  if (normalized == null) return 'N/A';
  return normalized.toFixed(1);
}

// Helper function to format phone number
function formatPhoneNumber(phone) {
  if (!phone) return 'N/A';

  const cleanPhone = phone.toString().replace('.0', '');

  if (cleanPhone.length >= 10) {
    const lastTen = cleanPhone.slice(-10);
    return `+91 ${lastTen.slice(0, 5)} ${lastTen.slice(5)}`;
  }

  return cleanPhone;
}

// Helper function to parse cost label
function parseCost(costStr) {
  if (!costStr) return 'On Request';

  const cost = String(costStr).replace(/[\u200B-\u200D\uFEFF]/g, '');

  if (cost.includes('₹')) {
    return cost;
  }

  const numMatch = cost.match(/\d+/);
  if (numMatch) {
    return `₹${numMatch[0]}`;
  }

  return cost;
}

function parseNumericCost(doctor) {
  const directFee = toNumber(doctor?.consultation_fee);
  if (directFee != null) return directFee;

  const costText = String(doctor?.cost || '').replace(/[^\d.]/g, '');
  const parsed = toNumber(costText);
  return parsed;
}

function haversineDistanceKm(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const earthRadiusKm = 6371;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const lat1Rad = toRad(lat1);
  const lat2Rad = toRad(lat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return earthRadiusKm * c;
}

function computeDoctorDistanceKm(doctor, userLocation) {
  const directDistance = toNumber(doctor?.distance_km);
  if (directDistance != null) return directDistance;

  const hospitalDistance = toNumber(doctor?.hospital_distance);
  if (hospitalDistance != null) return hospitalDistance;

  const doctorLat = toNumber(doctor?.latitude);
  const doctorLon = toNumber(doctor?.longitude);
  const userLat = toNumber(userLocation?.latitude);
  const userLon = toNumber(userLocation?.longitude);

  if (doctorLat == null || doctorLon == null || userLat == null || userLon == null) {
    return null;
  }

  return haversineDistanceKm(userLat, userLon, doctorLat, doctorLon);
}

function compareNullableAsc(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return a - b;
}

export default function DoctorList({ hospitals = [], doctors = [], specialty, user_location = null }) {
  const [sortBy, setSortBy] = useState('relevance');

  // Extract all doctors from all hospitals
  const doctorsFromHospitals = [];
  hospitals.forEach((hospital) => {
    if (hospital.doctors && Array.isArray(hospital.doctors)) {
      hospital.doctors.forEach((doctor) => {
        doctorsFromHospitals.push({
          ...doctor,
          hospital_name: hospital.name,
          hospital_distance: hospital.distance_km,
          hospital_rating: hospital.rating,
          hospital_cost: hospital.cost_level,
          hospital_insurance: hospital.insurance_supported,
        });
      });
    }
  });

  const sourceDoctors = doctors.length > 0 ? doctors : doctorsFromHospitals;
  const seenDoctors = new Set();
  const allDoctors = sourceDoctors.filter((doctor) => {
    const name = String(doctor?.name || '')
      .trim()
      .toLowerCase()
      .replace(/\s+/g, ' ');
    if (!name) return false;

    // Prefer unique display names so merged recommendation sources don't repeat cards.
    if (seenDoctors.has(name)) return false;
    seenDoctors.add(name);
    return true;
  });

  const userLocationKey = String(user_location?.city || '').trim().toLowerCase();

  const preparedDoctors = useMemo(
    () =>
      allDoctors.map((doctor) => {
        const distanceKm = computeDoctorDistanceKm(doctor, user_location);
        const ratingOutOfTen = normalizeDoctorRatingToTen(doctor?.rating ?? doctor?.rating_score);
        const feeValue = parseNumericCost(doctor);
        const resolvedLocation = getDoctorLocationText(doctor);
        const doctorLocation = resolvedLocation.toLowerCase();

        return {
          ...doctor,
          location: resolvedLocation,
          _distanceKm: distanceKm == null ? null : Number(distanceKm.toFixed(1)),
          _ratingOutOfTen: ratingOutOfTen == null ? null : Number(ratingOutOfTen.toFixed(1)),
          _feeValue: feeValue,
          _isExactLocation: Boolean(userLocationKey) && doctorLocation === userLocationKey,
          _isAvailableToday: Boolean(doctor?.availability?.includes('Available Today')),
        };
      }),
    [allDoctors, user_location, userLocationKey]
  );

  const sortedDoctors = useMemo(() => {
    const doctorsCopy = [...preparedDoctors];

    doctorsCopy.sort((a, b) => {
      if (sortBy === 'closest') {
        const distanceCompare = compareNullableAsc(a._distanceKm, b._distanceKm);
        if (distanceCompare !== 0) return distanceCompare;
      } else if (sortBy === 'cost') {
        const costCompare = compareNullableAsc(a._feeValue, b._feeValue);
        if (costCompare !== 0) return costCompare;
      } else if (sortBy === 'rating') {
        const aRating = a._ratingOutOfTen ?? -1;
        const bRating = b._ratingOutOfTen ?? -1;
        if (bRating !== aRating) return bRating - aRating;
      }

      if (sortBy === 'relevance') {
        const exactCompare = Number(b._isExactLocation) - Number(a._isExactLocation);
        if (exactCompare !== 0) return exactCompare;
      }

      const availabilityCompare = Number(b._isAvailableToday) - Number(a._isAvailableToday);
      if (availabilityCompare !== 0) return availabilityCompare;

      const ratingCompare = (b._ratingOutOfTen ?? -1) - (a._ratingOutOfTen ?? -1);
      if (ratingCompare !== 0) return ratingCompare;

      const costCompare = compareNullableAsc(a._feeValue, b._feeValue);
      if (costCompare !== 0) return costCompare;

      return compareNullableAsc(a._distanceKm, b._distanceKm);
    });

    return doctorsCopy;
  }, [preparedDoctors, sortBy]);

  if (sortedDoctors.length === 0) {
    return (
      <div className="doctor-list-container">
        <div className="empty-state">
          <p>No doctors available for {specialty}.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="doctor-list-container">
      <div className="doctors-header">
        <div>
          <h2 className="doctors-main-title">👨‍⚕️ Available Doctors</h2>
          <p className="doctors-subtitle">{sortedDoctors.length} specialists found</p>
        </div>

        <div className="doctor-controls">
          <label htmlFor="doctor-sort">Sort by</label>
          <select id="doctor-sort" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="relevance">Most Relevant</option>
            <option value="closest">Closest</option>
            <option value="cost">Lowest Cost</option>
            <option value="rating">Highest Rating</option>
          </select>
        </div>
      </div>

      <div className="doctors-grid">
        {sortedDoctors.map((doctor, index) => {
          const doctorPhone = doctor.phone || doctor.contact_number;
          return (
            <div key={doctor?.id || `${doctor?.name || 'doctor'}-${index}`} className="doctor-list-card">
              <div className="doctor-card-header">
                <div className="doctor-info-top">
                  <h3 className="doctor-list-name">{doctor.name}</h3>
                  <span
                    className={`availability-badge ${doctor.availability?.includes('Available Today') ? 'available' : 'scheduled'}`}
                  >
                    {doctor.availability || 'Contact for availability'}
                  </span>
                </div>
                <p className="doctor-list-specialty">👨‍⚕️ {doctor.specialty}</p>
              </div>

              <div className="doctor-card-body">
                <div className="doctor-stats">
                  <div className="stat-item">
                    <span className="stat-label">Experience</span>
                    <span className="stat-value">
                      {doctor.experience_years != null ? `${doctor.experience_years} yrs` : 'N/A'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Rating</span>
                    <span className="stat-value">{formatDoctorRatingOutOfTen(doctor._ratingOutOfTen)} / 10</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Per Visit</span>
                    <span className="stat-value">{parseCost(doctor.cost)}</span>
                  </div>
                </div>

                {doctor.hospital_name && (
                  <div className="hospital-info-section">
                    <p className="hospital-info-label">🏥 Hospital</p>
                    <p className="hospital-info-name">{doctor.hospital_name}</p>
                    <div className="hospital-metrics">
                      <span className="metric">
                        📍 {doctor.hospital_distance ?? 'N/A'}{' '}
                        {Number.isFinite(Number(doctor.hospital_distance)) ? 'km away' : ''}
                      </span>
                      <span className="metric">Hospital Rating: {doctor.hospital_rating ?? 'N/A'}/5</span>
                      {doctor.hospital_cost && (
                        <span className={`metric cost-${doctor.hospital_cost}`}>
                          {doctor.hospital_cost.charAt(0).toUpperCase() + doctor.hospital_cost.slice(1)}
                        </span>
                      )}
                    </div>
                    {doctor.hospital_insurance && <span className="insurance-tag">✓ Insurance Accepted</span>}
                  </div>
                )}

                {doctor.location && (
                  <div className="location-info">
                    <span className="location-label">🏢 Floor/Location</span>
                    <span className="location-value">{doctor.location}</span>
                    {doctor._distanceKm != null && (
                      <span className="location-distance">Approx. {doctor._distanceKm.toFixed(1)} km away</span>
                    )}
                  </div>
                )}

                {doctorPhone && doctorPhone !== 'Not available' && (
                  <div className="contact-info">
                    <span className="contact-label">📞 Call Doctor</span>
                    <span className="contact-value">{formatPhoneNumber(doctorPhone)}</span>
                  </div>
                )}
              </div>

              <div className="doctor-card-footer">
                <div style={{flex: 1}}>
                  <ReviewSummary doctorName={doctor.name} facilityName={doctor.hospital_name} />
                </div>
                <div style={{display: 'flex', gap: '8px'}}>
                  <button className="book-button">Book Appointment</button>
                  <button className="review-button">Review</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
