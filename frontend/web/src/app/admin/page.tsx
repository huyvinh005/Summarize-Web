"use client";

import { AuthResponse, fetchAdminKpis, fetchAdminMe, fetchAdminRecent, fetchAdminSeries } from "@/lib/api";
import { readPersistedAuth, readPersistedLocale } from "@/lib/auth";
import { dictionary, Locale } from "@/lib/content";
import { LoaderCircle } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

function AdminGate({ locale }: { locale: Locale }) {
  const t = dictionary[locale];

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(120,119,198,0.12),_transparent_35%),linear-gradient(180deg,#0b1020_0%,#0f172a_100%)] text-white">
      <section className="mx-auto flex min-h-screen w-full max-w-7xl items-center justify-center px-4 py-10 sm:px-6 lg:px-8">
        <div className="w-full max-w-lg rounded-[32px] border border-white/10 bg-white/6 p-4 shadow-[0_32px_100px_rgba(2,6,23,0.34)] backdrop-blur sm:p-5">
          <div className="rounded-[28px] border border-white/10 bg-slate-950/58 p-8 text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white/80">
              <LoaderCircle className="h-6 w-6 animate-spin" />
            </div>
            <p className="mt-5 text-sm text-white/50">{locale === "vi" ? "Đang kiểm tra quyền truy cập" : "Checking access"}</p>
            <h1 className="mt-2 text-2xl font-semibold text-white">{locale === "vi" ? "Đang mở admin" : "Opening admin"}</h1>
            <p className="mt-3 text-sm leading-7 text-white/65">{t.dashboardTitle}</p>
          </div>
        </div>
      </section>
    </main>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_18px_60px_rgba(2,6,23,0.28)]">
      <p className="text-xs font-medium uppercase tracking-wide text-white/50">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function SimpleBars({ points }: { points: { date: string; count: number }[] }) {
  const max = Math.max(1, ...points.map((p) => p.count));
  const shown = points.slice(-14);

  return (
    <div className="flex items-end gap-1.5">
      {shown.map((p) => (
        <div key={p.date} className="flex h-20 w-full items-end">
          <div
            className="w-full rounded-t-md bg-gradient-to-t from-indigo-500/40 to-indigo-200/30"
            style={{ height: `${Math.max(6, Math.round((p.count / max) * 100))}%` }}
            title={`${p.date}: ${p.count}`}
          />
        </div>
      ))}
    </div>
  );
}

export default function AdminPage() {
  const router = useRouter();
  const [locale, setLocale] = useState<Locale>("vi");
  const [auth, setAuth] = useState<AuthResponse | null>(null);
  const [isHydratingAuth, setIsHydratingAuth] = useState(true);
  const gateLocale = useMemo(() => locale, [locale]);

  const [kpis, setKpis] = useState<Awaited<ReturnType<typeof fetchAdminKpis>> | null>(null);
  const [series, setSeries] = useState<Awaited<ReturnType<typeof fetchAdminSeries>> | null>(null);
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof fetchAdminRecent>> | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const persisted = readPersistedAuth();
      setLocale(readPersistedLocale() ?? "vi");

      if (!persisted?.access_token) {
        setIsHydratingAuth(false);
        router.replace("/#auth");
        return;
      }

      setAuth(persisted);
      setIsHydratingAuth(false);
    });

    return () => window.cancelAnimationFrame(frame);
  }, [router]);

  useEffect(() => {
    const token = auth?.access_token;
    if (!token) return;
    const accessToken = token;

    let cancelled = false;
    async function load() {
      setIsLoadingData(true);
      try {
        await fetchAdminMe(accessToken);
        const [nextKpis, nextSeries, nextRecent] = await Promise.all([
          fetchAdminKpis(accessToken),
          fetchAdminSeries(accessToken, 30),
          fetchAdminRecent(accessToken, 10),
        ]);
        if (cancelled) return;
        setKpis(nextKpis);
        setSeries(nextSeries);
        setRecent(nextRecent);
      } catch (error) {
        if (cancelled) return;
        if (String(error).toLowerCase().includes("admin access required")) {
          router.replace("/dashboard");
          return;
        }
        router.replace("/dashboard");
      } finally {
        if (!cancelled) setIsLoadingData(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [auth?.access_token, router]);

  if (isHydratingAuth) {
    return <AdminGate locale={gateLocale} />;
  }

  if (!auth) {
    return null;
  }

  const t = dictionary[locale];

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(120,119,198,0.12),_transparent_35%),linear-gradient(180deg,#0b1020_0%,#0f172a_100%)] text-white">
      <section className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm text-white/55">{locale === "vi" ? "Dashboard nội bộ" : "Internal dashboard"}</p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">{locale === "vi" ? "Admin" : "Admin"}</h1>
            <p className="mt-2 text-sm text-white/60">{t.dashboardTitle}</p>
          </div>
          <button
            type="button"
            onClick={() => router.push("/dashboard")}
            className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/12 bg-white/[0.06] px-4 py-2.5 text-sm font-medium text-white/90 shadow-[0_14px_30px_rgba(2,6,23,0.36)] transition hover:-translate-y-0.5 hover:border-white/18 hover:bg-white/[0.09] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/30"
          >
            {locale === "vi" ? "Về dashboard" : "Back to dashboard"}
          </button>
        </header>

        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <MetricCard label={locale === "vi" ? "Người dùng" : "Users"} value={kpis ? String(kpis.total_users) : "—"} />
          <MetricCard
            label={locale === "vi" ? "Đã xác thực" : "Verified"}
            value={kpis ? String(kpis.verified_users) : "—"}
          />
          <MetricCard
            label={locale === "vi" ? "Tài liệu" : "Documents"}
            value={kpis ? String(kpis.total_documents) : "—"}
          />
          <MetricCard
            label={locale === "vi" ? "Bản tóm tắt" : "Summaries"}
            value={kpis ? String(kpis.total_summaries) : "—"}
          />
          <MetricCard
            label={locale === "vi" ? "Tóm tắt hôm nay" : "Today"}
            value={kpis ? String(kpis.summaries_today) : "—"}
          />
          <MetricCard
            label={locale === "vi" ? "7 ngày" : "7 days"}
            value={kpis ? String(kpis.summaries_7d) : "—"}
          />
        </div>

        <div className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-[28px] border border-white/10 bg-slate-950/40 p-6">
            <p className="text-sm font-semibold text-white">{locale === "vi" ? "Đăng ký (14 ngày)" : "Signups (14d)"}</p>
            <p className="mt-1 text-xs text-white/55">{locale === "vi" ? "Hiển thị 14 ngày gần nhất" : "Last 14 days"}</p>
            <div className="mt-4">{series ? <SimpleBars points={series.users} /> : <div className="h-20" />}</div>
          </div>
          <div className="rounded-[28px] border border-white/10 bg-slate-950/40 p-6">
            <p className="text-sm font-semibold text-white">{locale === "vi" ? "Tóm tắt (14 ngày)" : "Summaries (14d)"}</p>
            <p className="mt-1 text-xs text-white/55">{locale === "vi" ? "Hiển thị 14 ngày gần nhất" : "Last 14 days"}</p>
            <div className="mt-4">{series ? <SimpleBars points={series.summaries} /> : <div className="h-20" />}</div>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="rounded-[28px] border border-white/10 bg-slate-950/40 p-6 lg:col-span-1">
            <p className="text-sm font-semibold text-white">{locale === "vi" ? "Người dùng mới" : "Recent users"}</p>
            <div className="mt-4 space-y-3">
              {(recent?.users ?? []).slice(0, 8).map((u) => (
                <div key={u.id} className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
                  <p className="text-sm text-white/90">{u.email}</p>
                  <p className="mt-0.5 text-xs text-white/55">
                    {(u.is_verified ? (locale === "vi" ? "Đã xác thực" : "Verified") : locale === "vi" ? "Chưa xác thực" : "Unverified") +
                      " · " +
                      new Date(u.created_at).toLocaleString()}
                  </p>
                </div>
              ))}
              {!recent && <div className="text-sm text-white/60">{isLoadingData ? "…" : "—"}</div>}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-slate-950/40 p-6 lg:col-span-1">
            <p className="text-sm font-semibold text-white">{locale === "vi" ? "Tài liệu mới" : "Recent documents"}</p>
            <div className="mt-4 space-y-3">
              {(recent?.documents ?? []).slice(0, 8).map((d) => (
                <div key={d.id} className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
                  <p className="text-sm text-white/90">{d.title}</p>
                  <p className="mt-0.5 text-xs text-white/55">
                    {(d.source_type === "pdf" ? "PDF" : locale === "vi" ? "Văn bản" : "Text") +
                      (d.original_filename ? ` · ${d.original_filename}` : "") +
                      " · " +
                      new Date(d.created_at).toLocaleString()}
                  </p>
                </div>
              ))}
              {!recent && <div className="text-sm text-white/60">{isLoadingData ? "…" : "—"}</div>}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-slate-950/40 p-6 lg:col-span-1">
            <p className="text-sm font-semibold text-white">{locale === "vi" ? "Tóm tắt mới" : "Recent summaries"}</p>
            <div className="mt-4 space-y-3">
              {(recent?.summaries ?? []).slice(0, 8).map((s) => (
                <div key={s.id} className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
                  <p className="text-sm text-white/90">{s.method}</p>
                  <p className="mt-0.5 text-xs text-white/55">
                    {(s.source ? s.source : "") +
                      " · " +
                      new Date(s.created_at).toLocaleString() +
                      (typeof s.rating_average === "number" ? ` · ★ ${s.rating_average.toFixed(2)}` : "")}
                  </p>
                </div>
              ))}
              {!recent && <div className="text-sm text-white/60">{isLoadingData ? "…" : "—"}</div>}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
