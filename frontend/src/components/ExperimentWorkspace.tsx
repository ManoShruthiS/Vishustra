import { type FormEvent, useState } from "react";

import { createExperiment, runExperiment } from "../api";
import type { ExperimentRecord } from "../types";

interface ExperimentWorkspaceProps {
  experiments: ExperimentRecord[];
  onExperimentUpdated: (experiment: ExperimentRecord) => void;
}

function formatDate(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(
    new Date(timestamp)
  );
}

function statusLabel(status: ExperimentRecord["status"]): string {
  return status.replace(/^./, (letter) => letter.toUpperCase());
}

function formatAccuracy(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(3);
}

function formatDuration(value: number | undefined): string {
  return value === undefined ? "—" : `${value.toFixed(1)} s`;
}

export function ExperimentWorkspace({ experiments, onExperimentUpdated }: ExperimentWorkspaceProps) {
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [runningId, setRunningId] = useState<string | null>(null);

  async function handleCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim()) {
      setError("Give the experiment a clear name before saving it.");
      return;
    }

    setError(null);
    setIsCreating(true);
    try {
      const experiment = await createExperiment({ name: name.trim(), hypothesis: hypothesis.trim() });
      onExperimentUpdated(experiment);
      setName("");
      setHypothesis("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The experiment could not be saved.");
    } finally {
      setIsCreating(false);
    }
  }

  async function handleRun(expId: string) {
    setError(null);
    setRunningId(expId);
    try {
      onExperimentUpdated(await runExperiment(expId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The experiment could not be run.");
    } finally {
      setRunningId(null);
    }
  }

  return (
    <section className="workspace-panel" aria-labelledby="workspace-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Laboratory 02</p>
          <h2 id="workspace-title">Experiment Workspace</h2>
        </div>
        <p>Record a hypothesis, preserve its setup, and run the experiment.</p>
      </div>

      <form className="experiment-form" onSubmit={handleCreate}>
        <label>
          <span>Experiment name</span>
          <input
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Four-head baseline"
            value={name}
          />
        </label>
        <label>
          <span>Hypothesis</span>
          <input
            onChange={(event) => setHypothesis(event.target.value)}
            placeholder="What are you testing?"
            value={hypothesis}
          />
        </label>
        <button className="secondary-button" disabled={isCreating} type="submit">
          {isCreating ? "Saving…" : "Create experiment"}
        </button>
      </form>

      {error ? <p className="form-error" role="alert">{error}</p> : null}
      {experiments.length === 0 ? (
        <p className="empty-state">No experiments yet. Start with one focused question.</p>
      ) : (
        <ul className="experiment-list">
          {experiments.map((experiment) => (
            <li key={experiment.exp_id}>
              <div>
                <p>{experiment.exp_id}</p>
                <strong>{experiment.name}</strong>
                <span>Created {formatDate(experiment.created_at)}</span>
                {experiment.result ? (
                  <dl className="result-summary">
                    <div>
                      <dt>Validation accuracy</dt>
                      <dd>{formatAccuracy(experiment.result.val_acc)}</dd>
                    </div>
                    <div>
                      <dt>Validation loss</dt>
                      <dd>{formatNumber(experiment.result.val_loss)}</dd>
                    </div>
                    <div>
                      <dt>Training time</dt>
                      <dd>{formatDuration(experiment.result.train_time_s)}</dd>
                    </div>
                  </dl>
                ) : null}
              </div>
              <div className="experiment-action">
                <span className={`status-pill status-${experiment.status}`}>{statusLabel(experiment.status)}</span>
                {experiment.status === "created" ? (
                  <button
                    className="text-button"
                    disabled={runningId === experiment.exp_id}
                    onClick={() => void handleRun(experiment.exp_id)}
                    type="button"
                  >
                    {runningId === experiment.exp_id ? "Running…" : "Run"}
                  </button>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
