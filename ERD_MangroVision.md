# MangroVision Database ERD

## Entity Relationship Diagram (Mermaid)

```mermaid
erDiagram
    users {
        INTEGER id PK
        TEXT full_name "NOT NULL"
        TEXT email UK "NOT NULL UNIQUE"
        TEXT role "DEFAULT planner"
        TEXT organization
        TEXT password_hash
        TEXT created_at "DEFAULT CURRENT_TIMESTAMP"
        TEXT last_login
    }

    analyses {
        INTEGER id PK
        INTEGER user_id FK "REFERENCES users(id)"
        TEXT image_name "NOT NULL"
        TEXT analyzed_at "NOT NULL"
        REAL center_lat
        REAL center_lon
        REAL altitude_m
        REAL gsd_cm
        REAL coverage_w_m
        REAL coverage_h_m
        REAL total_area_m2
        INTEGER canopy_count
        INTEGER polygon_count
        REAL danger_area_m2
        REAL danger_pct
        REAL plantable_area_m2
        REAL plantable_pct
        INTEGER hexagon_count
        REAL ai_confidence
        REAL canopy_buffer_m
        REAL hexagon_size_m
        INTEGER forbidden_filtered
        INTEGER eroded_filtered
    }

    planting_points {
        INTEGER id PK
        INTEGER analysis_id FK "REFERENCES analyses(id) ON DELETE CASCADE"
        INTEGER point_num "NOT NULL"
        REAL latitude "NOT NULL"
        REAL longitude "NOT NULL"
        INTEGER pixel_x
        INTEGER pixel_y
        REAL buffer_m
        REAL area_m2
        TEXT status "CHECK (planned|planted|skipped) DEFAULT planned"
    }

    exclusion_zones {
        INTEGER id PK
        INTEGER user_id FK "REFERENCES users(id)"
        TEXT zone_type "NOT NULL CHECK (forbidden|eroded)"
        TEXT geometry_geojson "NOT NULL"
        TEXT reason
        TEXT created_at "DEFAULT CURRENT_TIMESTAMP"
    }

    users ||--o{ analyses : "runs"
    users ||--o{ exclusion_zones : "creates"
    analyses ||--o{ planting_points : "generates"
```

## Table Descriptions

| Table | Purpose | Records |
|-------|---------|---------|
| **users** | Tracks planner identity, organization, and last login time | 1 default ("MangroVision Planner") |
| **analyses** | Stores per-image analysis results and parameters | 1 per drone image analyzed |
| **planting_points** | GPS coordinates of recommended planting locations | ~3-50 per analysis |
| **exclusion_zones** | Forbidden/eroded areas stored as GeoJSON polygons | User-defined |

## Relationships

- **users -> analyses**: One user runs many analyses (1:N)
- **users -> exclusion_zones**: One user creates many exclusion zones (1:N)
- **analyses -> planting_points**: One analysis generates many planting points (1:N, CASCADE DELETE)

## Normalization

This schema follows **Third Normal Form (3NF)**:
- All attributes depend on the primary key (1NF)
- No partial dependencies on composite keys (2NF)
- No transitive dependencies between non-key attributes (3NF)
