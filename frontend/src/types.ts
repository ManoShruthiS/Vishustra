export interface HealthStatus {
  status: string;
  service: string;
  version: string;
}

export interface PaperConfig {
  name: string;
  num_encoder_layers: number;
  num_decoder_layers: number;
  d_model: number;
  d_ff: number;
  num_heads: number;
  d_k: number;
  d_v: number;
  dropout: number;
  label_smoothing: number;
  train_steps: number;
  params_millions: number;
  notes: string[];
}

export interface AttentionResult {
  tokens: string[];
  d_k: number;
  scaler: number;
  scores: number[][];
  scaled_scores: number[][];
  weights: number[][];
  output: number[][];
}

export interface ExperimentRecord {
  exp_id: string;
  name: string;
  status: "created" | "running" | "completed" | "failed";
  path: string;
  created_at: string;
  updated_at: string;
  config: Record<string, unknown>;
  notes: string[];
  result: Record<string, number> | null;
}

export interface CreateExperimentInput {
  name: string;
  hypothesis?: string;
  notes?: string;
  seed?: number;
  dataset?: Record<string, unknown>;
  architecture?: Record<string, unknown>;
  training?: Record<string, unknown>;
}
