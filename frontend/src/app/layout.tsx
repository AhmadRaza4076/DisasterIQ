import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DisasterIQ | Satellite Damage Triage — DarkNem",
  description:
    "Pakistan-focused satellite building damage triage for emergency coordinators. AMD Hackathon ACT II.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
