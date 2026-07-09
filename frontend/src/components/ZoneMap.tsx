"use client";

import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import type { AnalysisResult } from "@/lib/api";
import "leaflet/dist/leaflet.css";

interface Props {
  analysis: AnalysisResult | null;
}

const RANK_COLORS: Record<number, string> = {
  1: "#ef4444",
  2: "#f97316",
  3: "#eab308",
  4: "#3b82f6",
  5: "#22c55e",
};

function FitBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (positions.length === 0) return;
    if (positions.length === 1) {
      map.setView(positions[0], 15);
      return;
    }
    const lats = positions.map((p) => p[0]);
    const lngs = positions.map((p) => p[1]);
    map.fitBounds([
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    ]);
  }, [map, positions]);
  return null;
}

export default function ZoneMap({ analysis }: Props) {
  const geoZones = useMemo(
    () =>
      (analysis?.zones ?? []).filter(
        (z) => z.centroid_lat != null && z.centroid_lng != null
      ),
    [analysis]
  );
  const positions = useMemo<[number, number][]>(
    () => geoZones.map((z) => [z.centroid_lat!, z.centroid_lng!] as [number, number]),
    [geoZones]
  );

  if (!analysis) return null;

  if (!analysis.geo_available) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4 text-sm text-slate-500">
        Geographic coordinates available for demo pairs with xBD metadata only.
      </div>
    );
  }

  if (geoZones.length === 0) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4 text-sm text-slate-500">
        Geo metadata present, but no zone centroids resolved for this pair.
      </div>
    );
  }

  const center = positions[0];

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-slate-300">Geo-Referenced Priority Zones</h3>
      <div className="h-80 rounded-lg overflow-hidden border border-slate-700">
        <MapContainer
          center={center}
          zoom={14}
          className="h-full w-full"
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds positions={positions} />
          {geoZones.map((z) => (
            <CircleMarker
              key={z.rank}
              center={[z.centroid_lat!, z.centroid_lng!]}
              radius={10 + (9 - Math.min(z.rank, 9)) * 2}
              pathOptions={{
                color: RANK_COLORS[z.rank] ?? "#94a3b8",
                fillColor: RANK_COLORS[z.rank] ?? "#94a3b8",
                fillOpacity: 0.85,
                weight: 2,
              }}
            >
              <Popup>
                <div className="text-sm space-y-1">
                  <p className="font-semibold">Zone #{z.rank}</p>
                  <p>Score: {z.priority_score}</p>
                  <p>
                    Destroyed: {z.building_counts.destroyed}, Major: {z.building_counts.major}
                  </p>
                  <p className="text-xs text-slate-600">
                    {z.centroid_lat!.toFixed(5)}, {z.centroid_lng!.toFixed(5)}
                  </p>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
