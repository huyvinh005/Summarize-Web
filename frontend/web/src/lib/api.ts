export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? "Request failed");
  }

  return response.json() as Promise<T>;
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  return parseResponse<T>(response);
}

export async function apiAuthedRequest<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  return apiRequest<T>(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });
}

export async function apiFormRequest<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });

  return parseResponse<T>(response);
}

export type DocumentResponse = {
  id: string;
  title: string;
  source_type: "text" | "pdf";
  language: "vi" | "en";
  created_at: string;
  extraction_method?: string | null;
  reused_existing?: boolean;
};

export type UploadDocumentResponse = {
  id: string;
  title: string;
  source_type: "pdf";
  language: "vi" | "en";
  created_at: string;
  extraction_method?: string | null;
  reused_existing?: boolean;
};

export type SummaryRatingValue = 1 | 2 | 3 | 4 | 5;

export type SummaryResponse = {
  summary_id: string;
  document_id: string;
  source: string;
  language: "vi" | "en";
  summary: string;
  method: string;
  rating_average: number;
  rating_count: number;
  current_user_rating?: number | null;
  created_at: string;
};

export type SummaryRateRequest = {
  rating: SummaryRatingValue;
};

export type SummaryRatingResponse = {
  summary: SummaryResponse;
  message: string;
  preferred_for_document: boolean;
};

export type SummaryHistoryItem = {
  id: string;
  title: string;
  source_type: "text" | "pdf";
  created_at: string;
  status: "ready" | "processing";
  has_summary: boolean;
  extraction_method?: string | null;
  summary_created_at?: string | null;
  summary_id?: string | null;
  rating_average: number;
  rating_count: number;
  current_user_rating?: number | null;
};

export type SummaryDetailResponse = {
  document: DocumentResponse;
  summary: SummaryResponse | null;
  latest_summary: SummaryResponse | null;
  preferred_summary: SummaryResponse | null;
  available_summaries: SummaryResponse[];
};

export type ChatResponse = {
  answer: string;
  method: string;
  source: string;
  created_at: string;
};

export type VerificationResponse = {
  message: string;
  delivery_mode: string;
  expires_at: string;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
    is_verified: boolean;
    created_at: string;
  };
};
