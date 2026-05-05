/**
 * API client — thin wrapper around fetch for calling the FastAPI backend.
 * Base URL defaults to localhost:8000 during development.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 15000;

/** Generic fetch helper with JSON handling and auth header. */
async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Backend did not respond. Check that FastAPI is running on port 8000.");
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || "Request failed");
  }

  return res.json() as Promise<T>;
}

/* -------------------------------------------------- */
/*  Auth                                              */
/* -------------------------------------------------- */
export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  id: string;
  username: string;
  email: string;
  created_at: string;
}

export function register(username: string, email: string, password: string) {
  return request<UserResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password }),
  });
}

export function login(email: string, password: string) {
  return request<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

/* -------------------------------------------------- */
/*  AI                                                */
/* -------------------------------------------------- */
export interface SummarizeResponse {
  session_id: string;
  summary: string;
}

export interface ChatResponse {
  reply: string;
}

export interface SessionListItem {
  id: string;
  title: string;
  created_at: string;
}

export interface SessionDetail {
  id: string;
  user_id: string;
  title: string;
  original_text: string;
  summary: string;
  chat_history: { role: string; content: string; timestamp: string }[];
  created_at: string;
  updated_at: string;
}

export function summarize(
  text: string,
  sessionId?: string,
  targetWords: number = 500
) {
  return request<SummarizeResponse>("/api/ai/summarize", {
    method: "POST",
    body: JSON.stringify({
      text,
      session_id: sessionId,
      target_words: targetWords,
    }),
  });
}

export function summarizePdf(file: File, targetWords: number = 500) {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("target_words", String(targetWords));

  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  return fetch(`${API_BASE}/api/ai/summarize-pdf`, {
    method: "POST",
    headers,
    body: formData,
  }).then(async (res) => {
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || "Request failed");
    }
    return res.json() as Promise<SummarizeResponse>;
  });
}

export function chat(sessionId: string, prompt: string) {
  return request<ChatResponse>("/api/ai/chat", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, prompt }),
  });
}

export function getSessions() {
  return request<SessionListItem[]>("/api/ai/sessions");
}

export function getSession(sessionId: string) {
  return request<SessionDetail>(`/api/ai/sessions/${sessionId}`);
}
