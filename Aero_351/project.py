from math import pi, sin, cos, atan2, sqrt
from datetime import datetime, timedelta, timezone

MU_EARTH = 398600.4418      # km^3/s^2
SEC_PER_DAY = 86400.0
R_EARTH = 6378.1363         # km

def tle_epoch_to_datetime_julian(epoch_field):
    """
    epoch_field: YYDDD.DDDDDDD (e.g., 25316.14507882 for 2025 day 316.14507882)
    Returns dict with UTC datetime, JD, MJD, and Julian centuries T (TT≈UTC here).
    """
    yy = int(epoch_field // 1000)
    doy = epoch_field - yy*1000
    year = 2000 + yy if yy < 57 else 1900 + yy   # TLE epoch rule

    day_int = int(doy)
    frac_day = doy - day_int
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int-1,
                                                               seconds=frac_day*SEC_PER_DAY)

    # Julian Date (UTC) – algorithm valid for Gregorian calendar dates
    y, m, d = dt.year, dt.month, dt.day
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    frac = (dt.hour + (dt.minute + (dt.second + dt.microsecond/1e6)/60.0)/60.0) / 24.0
    JD = int(365.25*(y + 4716)) + int(30.6001*(m + 1)) + d + B - 1524.5 + frac
    MJD = JD - 2400000.5
    T = (JD - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    return {"datetime_utc": dt, "JD": JD, "MJD": MJD, "T_J2000": T}

def tle_fields_to_coe(inc_deg, raan_deg, e, argp_deg, M_deg, n_rev_per_day):
    n = n_rev_per_day * 2*pi / SEC_PER_DAY                # rad/s
    a = (MU_EARTH / (n*n))**(1/3)                         # km

    # Kepler solve
    M = (M_deg * pi/180.0 + pi) % (2*pi) - pi
    E = M if e < 0.8 else pi
    for _ in range(50):
        f = E - e*sin(E) - M
        fp = 1 - e*cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < 1e-12:
            break

    sin_v = sqrt(1-e*e) * sin(E) / (1 - e*cos(E))
    cos_v = (cos(E) - e)       / (1 - e*cos(E))
    v = atan2(sin_v, cos_v)
    v_deg = (v * 180.0/pi) % 360.0

    rp = a*(1 - e); ra = a*(1 + e)
    return {
        "a_km": a,
        "e": e,
        "i_deg": inc_deg,
        "raan_deg": raan_deg,
        "argp_deg": argp_deg,
        "M_deg": M_deg,
        "true_anomaly_deg": v_deg,
        "n_rev_per_day": n_rev_per_day,
        "rp_km": rp, "ra_km": ra,
        "hp_km": rp - R_EARTH, "ha_km": ra - R_EARTH
    }

# Example with your numbers
epoch_field = 25316.14507882
epoch_info = tle_epoch_to_datetime_julian(epoch_field)
coe = tle_fields_to_coe(0.9979, 85.7856, 0.0009167, 252.6559, 21.5383, 1.00437626)

print(epoch_info)
print(coe)
