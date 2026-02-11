from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd

app = FastAPI(title="VisionSafe Road Risk API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",  # local testing
        "https://lemon-ground-053dab000.2.azurestaticapps.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 1️⃣ Load data ----------

# segments table (unique segments)
segment_table = pd.read_csv("segment_table.csv")

# events JSON
with open("events_data.json", "r") as f:
    events_data = json.load(f)


def map_danger_category(d):
    if d >= 90:
        return "Extreme"
    elif d >= 75:
        return "High"
    elif d >= 50:
        return "Medium"
    elif d >= 25:
        return "Low"
    else:
        return "Very Low"


# ---------- Helper: compute segment-level risk ----------

def compute_segment_risk(filtered_events):
    df = pd.DataFrame(filtered_events)

    expected_cols = [
        "segment_id",
        "total_events",
        "total_risk_score",
        "avg_event_risk",
        "killed_count",
        "injured_count",
        "unknown_count",
        "danger_index",
        "danger_category",
    ]

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # aggregate per segment
    segment_scores = (
        df.groupby("segment_id")
        .agg(
            total_events=("event_id","count"),
            total_risk_score=("event_risk_score","sum"),
            avg_event_risk=("event_risk_score","mean"),
            killed_count=("severity", lambda x: (x=="killed").sum()),
            injured_count=("severity", lambda x: (x=="injured").sum())
        )
        .reset_index()
    )

    # danger_index normalization
    if not segment_scores.empty:
        # percentile-based danger score
        segment_scores["danger_index"] = (
                segment_scores["total_risk_score"]
                .rank(pct=True) * 100
        )

        segment_scores["unknown_count"] = segment_scores["total_events"] - segment_scores["killed_count"] - segment_scores["injured_count"]
        segment_scores["danger_category"] = segment_scores["danger_index"].apply(map_danger_category)

    return segment_scores


# ----------  Endpoints ----------
@app.get("/segments")
def get_segments(
    road_name: str | None = Query(None),
    time_of_day: str | None = Query(None),
    species: str | None = Query(None)
):
    filtered_events = events_data

    # Filter segments by road name if given
    if road_name:
        matching_segments = segment_table[
            segment_table["road_name"].str.contains(road_name, case=False, na=False)
        ]
        if matching_segments.empty:
            return {"segments": []}
    else:
        matching_segments = segment_table

    segment_ids = matching_segments["segment_id"].tolist()
    filtered_events = [e for e in filtered_events if e["segment_id"] in segment_ids]

    if time_of_day:
        filtered_events = [
            e for e in filtered_events
            if e["time_of_day"].lower() == time_of_day.lower()
        ]

    if species:
        filtered_events = [
            e for e in filtered_events
            if species.lower() in str(e.get("species","")).lower()
        ]

    # ---- compute risk only on matching events ----
    segment_scores = compute_segment_risk(filtered_events)

    # ---- attach segment info (not all segments) ----
    merged = matching_segments.merge(
        segment_scores,
        on="segment_id",
        how="left"
    )

    # 6️⃣ Fill missing events with zeros
    merged = merged.fillna({
        "total_events": 0,
        "total_risk_score": 0,
        "avg_event_risk": 0,
        "killed_count": 0,
        "injured_count": 0,
        "unknown_count": 0,
        "danger_index": 0,
        "danger_category": "Very Low"
    })

    api_data = {"segments": []}
    for _, row in merged.iterrows():
        api_data["segments"].append({
            "segment_id": int(row["segment_id"]),
            "road_name": row["road_name"],
            "district": row["district"],
            "region": row["region"],
            "start_lat": float(row["start_lat"]),
            "start_lon": float(row["start_lon"]),
            "total_events": int(row["total_events"]),
            "total_risk_score": float(row["total_risk_score"]),
            "avg_event_risk": float(row["avg_event_risk"]),
            "danger_index": float(row["danger_index"]),
            "danger_category": row["danger_category"],
            "event_breakdown": {
                "killed": int(row["killed_count"]),
                "injured": int(row["injured_count"]),
                "unknown": int(row["unknown_count"])
            }
        })

    return api_data


@app.get("/hotspots")
def get_hotspots(road_name: str | None = None, time_of_day: str | None = None, species: str | None = None):
    all_segments = get_segments(road_name=road_name, time_of_day=time_of_day, species=species)
    # filter by danger_category
    hotspots = [s for s in all_segments["segments"] if s.get("danger_category", "").lower() in ["high", "extreme"]]
    return {"segments": hotspots}
