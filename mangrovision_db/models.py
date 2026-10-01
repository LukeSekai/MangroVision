"""SQLAlchemy 2.x models for the core production persistence entities.

The initial Alembic baseline is intentionally explicit SQL because it imports a
large legacy schema in one maintenance window. These typed models are the
application-facing foundation for incremental repositories and future Alembic
autogeneration; spatial columns match the baseline exactly.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, Computed, Date, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    full_name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text, unique=True)
    role: Mapped[str] = mapped_column(Text, default="planner")
    organization: Mapped[str | None] = mapped_column(Text)
    password_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Organization(Base):
    __tablename__ = "organizations"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    normalized_name: Mapped[str] = mapped_column(Text, unique=True)
    inspection_interval_days: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProjectSite(Base):
    __tablename__ = "project_sites"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    tide_calibration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    polygon_geojson: Mapped[dict[str, Any]] = mapped_column(JSONB)
    geometry: Mapped[Any] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=False),
        Computed(
            "extensions.ST_Multi(extensions.ST_SetSRID("
            "extensions.ST_GeomFromGeoJSON(polygon_geojson::text), 4326))",
            persisted=True,
        ),
    )
    organization_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("mangrovision.organizations.id", ondelete="RESTRICT")
    )
    inspection_interval_days: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("mangrovision.users.id", ondelete="SET NULL")
    )
    image_name: Mapped[str] = mapped_column(Text)
    analysis_number: Mapped[int] = mapped_column(
        BigInteger, unique=True, server_default=text("nextval('mangrovision.analysis_number_seq')")
    )
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    center_lat: Mapped[float | None] = mapped_column(Float)
    center_lon: Mapped[float | None] = mapped_column(Float)
    center_location: Mapped[Any | None] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False),
        Computed(
            "CASE WHEN center_lat IS NULL OR center_lon IS NULL THEN NULL ELSE "
            "extensions.ST_SetSRID(extensions.ST_MakePoint(center_lon, center_lat), 4326) END",
            persisted=True,
        ),
    )
    project_site_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("mangrovision.project_sites.id", ondelete="SET NULL")
    )
    footprint_geojson: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    footprint: Mapped[Any | None] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=False),
        Computed(
            "CASE WHEN footprint_geojson IS NULL THEN NULL ELSE "
            "extensions.ST_Multi(extensions.ST_SetSRID("
            "extensions.ST_GeomFromGeoJSON(footprint_geojson::text), 4326)) END",
            persisted=True,
        ),
    )
    analysis_detail_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PlantingPoint(Base):
    __tablename__ = "planting_points"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("mangrovision.analyses.id", ondelete="CASCADE")
    )
    point_num: Mapped[int] = mapped_column(Integer)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    location: Mapped[Any] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False),
        Computed(
            "extensions.ST_SetSRID(extensions.ST_MakePoint(longitude, latitude), 4326)",
            persisted=True,
        ),
    )
    status: Mapped[str] = mapped_column(Text, default="planned")
    planted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    planted_date: Mapped[date | None] = mapped_column(Date)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MapZone(Base):
    __tablename__ = "map_zones"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    zone_type: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    warning_type: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    polygon_geojson: Mapped[dict[str, Any]] = mapped_column(JSONB)
    geometry: Mapped[Any] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=False),
        Computed(
            "extensions.ST_Multi(extensions.ST_SetSRID("
            "extensions.ST_GeomFromGeoJSON(polygon_geojson::text), 4326))",
            persisted=True,
        ),
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("mangrovision.users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnalysisAsset(Base):
    __tablename__ = "analysis_assets"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("mangrovision.analyses.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(Text)
    object_key: Mapped[str] = mapped_column(Text, unique=True)
    content_type: Mapped[str] = mapped_column(Text)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    lifecycle_state: Mapped[str] = mapped_column(Text, default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject_type: Mapped[str] = mapped_column(Text)
    subject_id: Mapped[int] = mapped_column(BigInteger)
    token_hash: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OrganizationMonitoringRecord(Base):
    __tablename__ = "organization_monitoring_records"
    __table_args__ = {"schema": "mangrovision"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("mangrovision.organizations.id", ondelete="RESTRICT")
    )
    monitored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    alive_count: Mapped[int] = mapped_column(Integer)
    dead_count: Mapped[int] = mapped_column(Integer)
    average_height_cm: Mapped[float] = mapped_column(Float)
    health_status: Mapped[str] = mapped_column(Text)
    actions_taken: Mapped[str] = mapped_column(Text)
    inspector_user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("mangrovision.users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
