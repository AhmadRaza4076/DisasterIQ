"use client";

import type { AnalysisResult } from "@/lib/api";

interface Props {
  analysis: AnalysisResult | null;
}

export default function ZoneTable({ analysis }: Props) {
  if (!analysis) return null;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Priority Zones</h3>
      <p className="text-xs text-slate-500 mb-3">Counts are distinct buildings (connected structures), not pixels.</p>
      <p className="text-xs text-slate-500 mb-3">Confidence shown for fine-tuned PyTorch mode; — in baseline/stub.</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 border-b border-slate-700">
              <th scope="col" className="pb-2 pr-4">Rank</th>
              <th scope="col" className="pb-2 pr-4">Score</th>
              <th scope="col" className="pb-2 pr-4">Confidence</th>
              <th scope="col" className="pb-2 pr-4">Destroyed</th>
              <th scope="col" className="pb-2 pr-4">Major</th>
              <th scope="col" className="pb-2 pr-4">Minor</th>
              <th scope="col" className="pb-2">OK</th>
            </tr>
          </thead>
          <tbody>
            {analysis.zones.slice(0, 8).map((z) => (
              <tr key={z.rank} className="border-b border-slate-800 text-slate-300">
                <td className="py-2 pr-4 font-medium">#{z.rank}</td>
                <td className="py-2 pr-4">{z.priority_score}</td>
                <td className="py-2 pr-4">
                  {z.confidence != null ? `${(z.confidence * 100).toFixed(0)}%` : "—"}
                </td>
                <td className="py-2 pr-4 text-red-400">{z.building_counts.destroyed}</td>
                <td className="py-2 pr-4 text-orange-400">{z.building_counts.major}</td>
                <td className="py-2 pr-4 text-blue-400">{z.building_counts.minor}</td>
                <td className="py-2 text-green-400">{z.building_counts.none}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-400">
        <div>Total buildings: {analysis.summary.total_buildings}</div>
        <div>Destroyed: {analysis.summary.destroyed_pct}%</div>
        <div>Major: {analysis.summary.major_pct}%</div>
        <div>Minor: {analysis.summary.minor_pct}%</div>
      </div>
    </div>
  );
}
