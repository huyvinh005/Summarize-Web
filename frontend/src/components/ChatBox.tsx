"use client";

import { useState, useRef, useEffect, type FormEvent } from "react";
import { Send, Bot, UserIcon, Loader2, MessageSquare } from "lucide-react";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatBoxProps {
  messages: ChatMessage[];
  onSend: (prompt: string) => void;
  loading: boolean;
  /** Whether a session is active (disable chat if not). */
  disabled: boolean;
}

export default function ChatBox({
  messages,
  onSend,
  loading,
  disabled,
}: ChatBoxProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading || disabled) return;
    onSend(input.trim());
    setInput("");
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-4 border-b border-slate-800">
        <MessageSquare className="w-5 h-5 text-brand-400" />
        <h2 className="text-sm font-semibold text-slate-200">AI Chat</h2>
      </div>

      {/* Messages Area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4"
      >
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="w-12 h-12 text-slate-700 mb-3" />
            <p className="text-sm text-slate-500">
              {disabled
                ? "Summarize a document first to start chatting"
                : "Ask questions about your document"}
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {/* AI avatar (left) */}
            {msg.role === "assistant" && (
              <div className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center bg-brand-600/20">
                <Bot className="w-4 h-4 text-brand-400" />
              </div>
            )}

            {/* Bubble */}
            <div
              className={`max-w-[75%] px-4 py-2.5 text-sm leading-relaxed rounded-2xl ${
                msg.role === "user"
                  ? "bg-brand-600 text-white rounded-br-md"
                  : "bg-surface-raised text-slate-300 border border-slate-700 rounded-bl-md"
              }`}
            >
              {msg.content}
            </div>

            {/* User avatar (right) */}
            {msg.role === "user" && (
              <div className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center bg-slate-700">
                <UserIcon className="w-4 h-4 text-slate-300" />
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center bg-brand-600/20">
              <Bot className="w-4 h-4 text-brand-400" />
            </div>
            <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-surface-raised border border-slate-700">
              <Loader2 className="w-4 h-4 animate-spin text-brand-400" />
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800 p-4 flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={disabled}
          placeholder={
            disabled
              ? "Summarize a document to enable chat"
              : "Ask something about the document..."
          }
          className="flex-1 px-4 py-2.5 rounded-xl text-sm bg-surface-raised border border-slate-700 text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading || disabled}
          className="px-4 py-2.5 rounded-xl transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-brand-600 hover:bg-brand-500 text-white"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
