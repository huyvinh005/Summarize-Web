"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import TextInput from "@/components/TextInput";
import ChatBox, { type ChatMessage } from "@/components/ChatBox";
import {
  summarize,
  summarizePdf,
  chat,
  getSessions,
  getSession,
  type SessionListItem,
} from "@/lib/api";

export default function DashboardPage() {
  const router = useRouter();

  // ---- State ----
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [summary, setSummary] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingSummarize, setLoadingSummarize] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);

  // ---- Auth guard ----
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    }
  }, [router]);

  // ---- Fetch session list ----
  const fetchSessions = useCallback(async () => {
    try {
      setLoadingSessions(true);
      const data = await getSessions();
      setSessions(data);
    } catch {
      // token invalid — redirect to login
      localStorage.removeItem("token");
      router.push("/login");
    } finally {
      setLoadingSessions(false);
    }
  }, [router]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // ---- Handlers ----
  function handleNewSession() {
    setActiveSessionId(null);
    setSummary("");
    setMessages([]);
  }

  async function handleSelectSession(id: string) {
    setActiveSessionId(id);
    try {
      const data = await getSession(id);
      setSummary(data.summary || "");
      setMessages(
        data.chat_history.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        }))
      );
    } catch {
      setSummary("");
      setMessages([]);
    }
  }

  async function handleSummarize(text: string, targetWords: number) {
    setLoadingSummarize(true);
    try {
      const data = await summarize(text, activeSessionId ?? undefined, targetWords);
      setSummary(data.summary);
      setActiveSessionId(data.session_id);
      // Refresh sidebar list
      await fetchSessions();
    } catch (err) {
      setSummary(
        err instanceof Error ? `Error: ${err.message}` : "Summarization failed"
      );
    } finally {
      setLoadingSummarize(false);
    }
  }

  async function handleSummarizePdf(file: File, targetWords: number) {
    setLoadingSummarize(true);
    try {
      const data = await summarizePdf(file, targetWords);
      setSummary(data.summary);
      setActiveSessionId(data.session_id);
      await fetchSessions();
    } catch (err) {
      setSummary(
        err instanceof Error ? `Error: ${err.message}` : "PDF summarization failed"
      );
    } finally {
      setLoadingSummarize(false);
    }
  }

  async function handleChat(prompt: string) {
    if (!activeSessionId) return;
    // Optimistically add user message
    setMessages((prev) => [...prev, { role: "user", content: prompt }]);
    setLoadingChat(true);
    try {
      const data = await chat(activeSessionId, prompt);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, something went wrong." },
      ]);
    } finally {
      setLoadingChat(false);
    }
  }

  function handleLogout() {
    localStorage.removeItem("token");
    router.push("/login");
  }

  // ---- Render ----
  return (
    <div className="flex h-screen bg-surface">
      {/* Left Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        loading={loadingSessions}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        onLogout={handleLogout}
      />

      {/* Main Content — 2 columns */}
      <main className="flex-1 flex min-w-0">
        {/* Left column — Document Input */}
        <div className="flex-1 border-r border-slate-800 min-w-0">
          <TextInput
            onSummarize={handleSummarize}
            onSummarizePdf={handleSummarizePdf}
            summary={summary}
            loading={loadingSummarize}
          />
        </div>

        {/* Right column — AI Chat */}
        <div className="flex-1 min-w-0">
          <ChatBox
            messages={messages}
            onSend={handleChat}
            loading={loadingChat}
            disabled={!activeSessionId}
          />
        </div>
      </main>
    </div>
  );
}
