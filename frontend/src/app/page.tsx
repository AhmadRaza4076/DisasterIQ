"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import BriefPanel from "@/components/BriefPanel";
import DamageCanvas from "@/components/DamageCanvas";
import ZoneTable from "@/components/ZoneTable";
import {
  analyzeDemoPair,
  analyzeUpload,
  demoImageUrl,
  fetchBrief,
  fetchDemoPairs,
  fetchFieldReport,
  fetchHealth,
  type AnalysisResult,
  type DemoPair,
  type HealthResponse,
} from "@/lib/api";

const ZoneMap = dynamic(() => import("@/components/ZoneMap"), { ssr: false });

const PAKISTAN_CONTEXT =
  "Pakistan disaster response: 2022 monsoon floods and earthquake-affected areas. " +
  "Prioritize zones with destroyed and major damage for rescue and relief deployment.";

export default function HomePage() {
  const [pairs, setPairs] = useState<DemoPair[]>([]);
  const [selectedPair, setSelectedPair] = useState<string>("");
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [postUrl, setPostUrl] = useState<string>("");
  const [preUrl, setPreUrl] = useState<string>("");
  const [brief, setBrief] = useState<string | null>(null);
  const [briefSource, setBriefSource] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [briefLoading, setBriefLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    fetchDemoPairs()
      .then((p) => {
        setPairs(p);
        if (p.length > 0) setSelectedPair(p[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    setBrief(null);
    setLoadingStage(
      health?.inference_mode === "docker"
        ? "Model inference (~2 min)…"
        : "Scoring damage zones…"
    );
    try {
      let result: AnalysisResult;
      if (preFile && postFile) {
        result = await analyzeUpload(preFile, postFile);
        setPreUrl(URL.createObjectURL(preFile));
        setPostUrl(URL.createObjectURL(postFile));
      } else if (selectedPair) {
        result = await analyzeDemoPair(selectedPair);
        const pair = pairs.find((p) => p.id === selectedPair);
        if (pair) {
          setPreUrl(demoImageUrl(pair.pre_image));
          setPostUrl(demoImageUrl(pair.post_image));
        }
      } else {
        throw new Error("Select a demo pair or upload images");
      }
      setAnalysis(result);

      setBriefLoading(true);
      setLoadingStage("Generating situation brief…");
      const briefResp = await fetchBrief(result, PAKISTAN_CONTEXT);
      setBrief(briefResp.brief);
      setBriefSource(briefResp.source);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      setBriefLoading(false);
      setLoadingStage(null);
    }
  }, [preFile, postFile, selectedPair, pairs, health?.inference_mode]);

  const downloadReport = useCallback(async () => {
    if (!analysis || !brief) return;
    setReportLoading(true);
    setError(null);
    try {
      const blob = await fetchFieldReport(analysis, brief);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `disasteriq-report-${analysis.pair_id || "analysis"}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    } finally {
      setReportLoading(false);
    }
  }, [analysis, brief]);

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <header className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs uppercase tracking-widest text-amber-500/90 font-medium">Team DarkNem</p>
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
            AMD ACT II · Unicorn
          </span>
          {health && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
              API {health.status} · {health.inference_mode} · {health.demo_pairs} pairs
            </span>
          )}
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
          DisasterIQ
        </h1>
        <p className="text-lg text-slate-300 max-w-3xl">
          Satellite triage for Pakistan emergency response — rank damaged zones from pre/post imagery so coordinators know where to send rescue teams first.
        </p>
        <p className="text-sm text-slate-500 max-w-3xl">
          ML scores building damage deterministically; the situation brief narrates ranked zones only (never re-orders priorities). Demo pairs use xBD earthquake and flood imagery as hazard proxies.
        </p>
      </header>

      <section className="rounded-lg border border-amber-900/30 bg-amber-950/20 px-4 py-3 text-sm text-amber-100/90">
        <strong className="text-amber-400">Mission context:</strong> 2022 Pakistan floods and seismic events displaced millions. Rapid zone-level damage maps help NDMA-style coordination when ground access is limited.
      </section>

      <section className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-4 rounded-lg border border-slate-700 bg-slate-900/40 p-4">
          <h2 className="font-semibold text-slate-200">Imagery input</h2>

          <label className="block text-sm text-slate-400">Demo pair (xBD — earthquake &amp; flood)</label>
          <select
            value={selectedPair}
            onChange={(e) => {
              setSelectedPair(e.target.value);
              setPreFile(null);
              setPostFile(null);
            }}
            className="w-full rounded bg-slate-800 border border-slate-600 px-3 py-2 text-sm text-slate-100"
            disabled={pairs.length === 0}
          >
            {pairs.length === 0 ? (
              <option value="">No demo pairs — run restore_demo_from_tar.ps1</option>
            ) : (
              pairs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.disaster_type}: {p.id}
                </option>
              ))
            )}
          </select>

          <div className="text-center text-xs text-slate-500">— or upload custom pair —</div>

          <label className="block text-sm text-slate-400">Pre-disaster image</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setPreFile(e.target.files?.[0] ?? null)}
            className="w-full text-sm text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-slate-700 file:px-2 file:py-1 file:text-slate-200"
          />

          <label className="block text-sm text-slate-400">Post-disaster image</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setPostFile(e.target.files?.[0] ?? null)}
            className="w-full text-sm text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-slate-700 file:px-2 file:py-1 file:text-slate-200"
          />

          <button
            onClick={runAnalysis}
            disabled={loading || (pairs.length === 0 && !(preFile && postFile))}
            className="w-full rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-50 px-4 py-2.5 font-medium text-white transition shadow-lg shadow-amber-900/20"
          >
            {loading ? (loadingStage ?? "Analyzing…") : "Analyze & brief"}
          </button>

          {loadingStage && (
            <p className="text-xs text-amber-400/90 animate-pulse">{loadingStage}</p>
          )}

          {error && <p className="text-red-400 text-sm">{error}</p>}

          {analysis && (
            <p className="text-xs text-slate-500">
              Inference: <span className="text-slate-400">{analysis.inference_mode}</span>
              {analysis.pair_id && (
                <> · Pair: <span className="text-slate-400">{analysis.pair_id}</span></>
              )}
            </p>
          )}

          <div className="text-xs text-slate-500 space-y-1 pt-2 border-t border-slate-800">
            <p className="font-medium text-slate-400">Damage legend</p>
            <p><span className="text-green-400">■</span> No damage</p>
            <p><span className="text-blue-400">■</span> Minor</p>
            <p><span className="text-orange-400">■</span> Major</p>
            <p><span className="text-red-400">■</span> Destroyed</p>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-slate-500 mb-1 uppercase tracking-wide">Pre-disaster</p>
              {preUrl ? (
                <img src={preUrl} alt="Pre disaster" className="rounded border border-slate-700 w-full" />
              ) : (
                <div className="h-48 rounded border border-dashed border-slate-700 flex items-center justify-center text-slate-600 text-sm">
                  Select a demo pair or upload images
                </div>
              )}
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1 uppercase tracking-wide">Post-disaster + damage overlay</p>
              {postUrl ? (
                <DamageCanvas postImageUrl={postUrl} analysis={analysis} />
              ) : (
                <div className="h-48 rounded border border-dashed border-slate-700 flex items-center justify-center text-slate-600 text-sm">
                  Overlay appears after analysis
                </div>
              )}
            </div>
          </div>

          <ZoneTable analysis={analysis} />
          <ZoneMap analysis={analysis} />
          <BriefPanel brief={brief} source={briefSource} loading={briefLoading} />
          {analysis && brief && (
            <button
              onClick={downloadReport}
              disabled={reportLoading}
              className="rounded-lg border border-slate-600 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 px-4 py-2.5 text-sm font-medium text-slate-200 transition"
            >
              {reportLoading ? "Generating PDF…" : "Download field report (PDF)"}
            </button>
          )}
        </div>
      </section>
    </main>
  );
}
