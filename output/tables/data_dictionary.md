# Data Dictionary — Emergency Healthcare Access Peru

## 1. IPRESS (ipress_clean.gpkg)

| Field | Type | Description |
|---|---|---|
| `codigo_unico` | str | Unique IPRESS identifier (CO_IPRESS) — join key with SUSALUD |
| `nombre` | str | Facility name |
| `institucion` | str | Governing sector: MINSA, ESSALUD, GOBIERNO REGIONAL, PRIVADO, OTRO |
| `clasificacion` | str | Facility class: HOSPITAL, CENTRO DE SALUD, PUESTO DE SALUD, CONSULTORIO, etc. |
| `tipo` | str | With or without inpatient beds |
| `categoria` | str | Service level: I-1, I-2, I-3, I-4, II-1, II-2, III-1, III-2, Sin Categoría |
| `ubigeo` | str(6) | 6-digit district code — primary join key across all datasets |
| `departamento` | str | Department name (uppercase) |
| `provincia` | str | Province name (uppercase) |
| `distrito` | str | District name (uppercase) |
| `longitud` | float | Decimal longitude in WGS-84 (raw column was labeled "NORTE" — mislabeled) |
| `latitud` | float | Decimal latitude in WGS-84 (raw column was labeled "ESTE" — mislabeled) |
| `altitud` | float | Elevation in meters above sea level |
| `camas` | int | Number of inpatient beds (0 for facilities without hospitalization) |
| `estado` | str | ACTIVADO / DESACTIVADO — only ACTIVADO kept in cleaned file |
| `condicion` | str | EN FUNCIONAMIENTO / CLAUSURADO |
| `coords_valid` | bool | True if coordinates fall within Peru's bounding box |
| `geometry` | Point | WGS-84 geometry (None for facilities without valid coordinates) |

**Source:** MINSA open data — https://www.datosabiertos.gob.pe/dataset/minsa-ipress  
**Cleaning decisions:**
- Kept only `estado == 'ACTIVADO'` (~700 deactivated facilities removed).
- Raw columns `NORTE` / `ESTE` store longitude / latitude values respectively — naming is inverted relative to standard Spanish usage; confirmed by cross-checking known district locations.
- Coordinates outside Peru bounding box (lon ∈ [−82, −68], lat ∈ [−19, 0]) flagged as invalid.

---

## 2. SUSALUD — Facility Annual (susalud_facility_annual.parquet)

| Field | Type | Description |
|---|---|---|
| `ubigeo` | str(6) | District code |
| `anho` | Int64 | Year (2015–2026) |
| `co_ipress` | str | IPRESS identifier — join key with IPRESS registry |
| `nombre` | str | Facility name as reported to SUSALUD |
| `sector` | str | MINSA, ESSALUD, GOBIERNO REGIONAL, PRIVADO, OTRO |
| `categoria` | str | Service level at time of reporting |
| `departamento` | str | Department |
| `provincia` | str | Province |
| `distrito` | str | District |
| `total_atenciones` | int | Total emergency consultations (sum over sex × age × month) |
| `total_atendidos` | int | Total unique patients attended |
| `meses_activo` | int | Number of calendar months with at least one reported record |

---

## 3. SUSALUD — District Annual (susalud_district_annual.parquet)

| Field | Type | Description |
|---|---|---|
| `ubigeo` | str(6) | District code — join key |
| `anho` | Int64 | Year |
| `departamento` | str | Department |
| `provincia` | str | Province |
| `distrito` | str | District |
| `total_atenciones` | int | Total district-level emergency consultations that year |
| `total_atendidos` | int | Total district-level patients attended that year |
| `n_ipress_reportantes` | int | Number of distinct facilities that reported activity that year |

**Source:** SUSALUD — http://datos.susalud.gob.pe (Consulta C1 — Producción Asistencial en Emergencia)  
**Cleaning decisions:**
- Rows where `total_atenciones ∈ {NE_0001, NE_0002}` are non-reporters (facility did not submit disaggregated data). Dropped before aggregation to avoid counting them as zeros.
- Data is originally disaggregated by sex × age × month. Aggregated upward to facility × year and district × year.
- UBIGEO zero-padded to 6 characters for all join operations.

---

## 4. CCPP (ccpp_clean.gpkg)

| Field | Type | Description |
|---|---|---|
| `ubigeo` | str(6) | District code where the populated center is located |
| `longitud` | float | Point longitude (WGS-84) |
| `latitud` | float | Point latitude (WGS-84) |
| `geometry` | Point | WGS-84 point geometry |
| *(other columns)* | varies | IGN attributes (population, name, type — preserved as-is) |

**Source:** IGN — Centros Poblados 1:100 000  
**Cleaning decisions:**
- Reprojected to WGS-84 (EPSG:4326) if source CRS differed.
- Rows with null or empty geometry removed.

---

## 5. DISTRITOS (distritos_clean.gpkg)

| Field | Type | Description |
|---|---|---|
| `ubigeo` | str(6) | 6-digit district code — primary join key |
| `iddpto` | str | Department code |
| `idprov` | str | Province code |
| `geometry` | Polygon | District boundary polygon (WGS-84) |

**Source:** INEI / d2cml-ai repository — DISTRITOS.shp  
**Cleaning decisions:**
- Reprojected to WGS-84 (EPSG:4326).
- Only identifier and geometry columns retained; all other attributes dropped to minimize file size.

---

## UBIGEO Reference

UBIGEO is a 6-digit code structured as:
```
DD PP III
│  │   └─ District (3 digits)
│  └───── Province (2 digits)
└──────── Department (2 digits)
```
Example: `150101` = Lima (15) / Lima (01) / Lima (01)

All datasets are joined on this key. Stored as zero-padded 6-character string.
