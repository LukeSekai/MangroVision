"""Current physical point status shared by the map and dashboard."""


def current_point_status(point: dict) -> str | None:
    if point.get("deleted_at") or point.get("is_deleted"):
        return None
    if point.get("death_at"):
        return "dead"
    planting = point.get("planting_status", point.get("point_status", point.get("status")))
    assignment = point.get("assignment_point_status", point.get("assignment_status"))
    if planting == "skipped" or assignment == "skipped":
        return "skipped"
    if planting == "planted" or assignment == "completed":
        return "planted"
    if point.get("eroded_unavailable") or point.get("inside_eroded_zone"):
        return "unavailable"
    if (
        assignment == "pending"
        or point.get("assignment_id") is not None
        or point.get("assigned_planter_id") is not None
        or point.get("assigned_planter_name")
    ):
        return "assigned"
    return "planned"
