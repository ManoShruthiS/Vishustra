import type {
  AttentionResult,
  CreateExperimentInput,
  ExperimentRecord,
  HealthStatus,
  PaperConfig
} from "./types";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init
  });

  if (!response.ok) {
    const error = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(error?.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

export function getPaperConfigs(): Promise<PaperConfig[]> {
  return request<PaperConfig[]>("/api/configs");
}

export function analyzeAttention(tokens: string[], d_k: number, seed: number): Promise<AttentionResult> {
  return request<AttentionResult>("/api/attention/analyze", {
    method: "POST",
    body: JSON.stringify({ tokens, d_k, seed })
  });
}

export function getExperiments(): Promise<ExperimentRecord[]> {
  return request<ExperimentRecord[]>("/api/experiments");
}

export function createExperiment(input: CreateExperimentInput): Promise<ExperimentRecord> {
  return request<ExperimentRecord>("/api/experiments", {
    method: "POST",
    body: JSON.stringify(input)
  });
}

export function runExperiment(expId: string): Promise<ExperimentRecord> {
  return request<ExperimentRecord>(`/api/experiments/${encodeURIComponent(expId)}/run`, {
    method: "POST"
  });
}
