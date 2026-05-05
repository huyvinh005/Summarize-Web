"use client";

import {
  PlusCircle,
  LogOut,
  FileText,
  Loader2,
} from "lucide-react";
import type { SessionListItem } from "@/lib/api";

interface SidebarProps {
  sessions: SessionListItem[];
  activeSessionId: string | null;
  loading: boolean;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
  onLogout: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  loading,
  onNewSession,
  onSelectSession,
  onLogout,
}: SidebarProps) {
  return (
    <aside className="flex flex-col w-64 h-screen shrink-0 bg-surface border-r border-slate-800">
      {/* Logo / Brand */}
      <div className="px-5 py-5 border-b border-slate-800">
        <h2 className="text-lg font-bold tracking-tight text-gradient">
          AI Summarizer
        </h2>
      </div>

      {/* New Session Button */}
      <div className="px-3 pt-4 pb-2">
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-white transition-all bg-brand-600 hover:bg-brand-500"
          style={{ boxShadow: "var(--shadow-glow)" }}
        >
          <PlusCircle className="w-4 h-4" />
          New Summary
        </button>
      </div>

      {/* Session History */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-3 py-2">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider px-2 mb-2">
          History
        </p>

        {loading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-slate-500" />
          </div>
        ) : sessions.length === 0 ? (
          <p className="text-xs text-slate-600 px-2 py-4">
            No sessions yet. Start a new summary!
          </p>
        ) : (
          <ul className="space-y-1">
            {sessions.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => onSelectSession(s.id)}
                  className={`w-full text-left flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all truncate ${
                    s.id === activeSessionId
                      ? "bg-surface-raised text-slate-100"
                      : "text-slate-400 hover:text-slate-200 hover:bg-surface-raised/50"
                  }`}
                >
                  <FileText className="w-4 h-4 shrink-0" />
                  <span className="truncate">{s.title}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Logout */}
      <div className="px-3 py-4 border-t border-slate-800">
        <button
          onClick={onLogout}
          className="w-full flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
        >
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </aside>
  );
}
