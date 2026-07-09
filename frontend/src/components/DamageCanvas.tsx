"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "@/lib/api";

interface Props {
  postImageUrl: string;
  analysis: AnalysisResult | null;
}

export default function DamageCanvas({ postImageUrl, analysis }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !postImageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;
    setLoaded(false);
    setFailed(false);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onerror = () => {
      if (!cancelled) setFailed(true);
    };
    img.onload = () => {
      if (cancelled) return;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      if (analysis?.mask_base64) {
        const overlay = new Image();
        overlay.onerror = () => {
          if (!cancelled) setFailed(true);
        };
        overlay.onload = () => {
          if (cancelled) return;
          ctx.drawImage(overlay, 0, 0);
          if (analysis.zones) {
            ctx.strokeStyle = "rgba(255,255,255,0.6)";
            ctx.lineWidth = 2;
            analysis.zones.slice(0, 8).forEach((z) => {
              const [x, y, w, h] = z.bbox;
              ctx.strokeRect(x, y, w, h);
              ctx.fillStyle = "rgba(255,255,255,0.9)";
              ctx.font = "bold 14px sans-serif";
              ctx.fillText(`#${z.rank}`, x + 4, y + 18);
            });
          }
          setLoaded(true);
        };
        overlay.src = `data:image/png;base64,${analysis.mask_base64}`;
      } else {
        setLoaded(true);
      }
    };
    img.src = postImageUrl;

    return () => {
      cancelled = true;
    };
  }, [postImageUrl, analysis]);

  return (
    <div className="relative rounded-lg overflow-hidden border border-slate-700 bg-slate-900">
      <canvas ref={canvasRef} className="max-w-full h-auto w-full" />
      {!loaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 text-slate-400">
          {failed ? "Failed to load imagery" : "Loading imagery..."}
        </div>
      )}
    </div>
  );
}
