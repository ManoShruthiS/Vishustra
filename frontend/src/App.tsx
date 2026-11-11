import { useEffect, useState } from "react";

import { getExperiments, getHealth, getPaperConfigs } from "./api";
import { AttentionLaboratory } from "./components/AttentionLaboratory";
import { ExperimentWorkspace } from "./components/ExperimentWorkspace";
import { MetricCard } from "./components/MetricCard";
import type { ExperimentRecord, HealthStatus, PaperConfig } from "./types";

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [configs, setConfigs] = useState<PaperConfig[]>([]);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isCurrent = true;

    async function loadWorkspace() {
      try {
        const [nextHealth, nextConfigs, nextExperiments] = await Promise.all([
          getHealth(),
          getPaperConfigs(),
          getExperiments()
        ]);
        if (isCurrent) {
          setHealth(nextHealth);
          setConfigs(nextConfigs);
          setExperiments(nextExperiments);
        }
      } catch (reason) {
        if (isCurrent) {
          setError(reason instanceof Error ? reason.message : "The VISHUSTRA API is unavailable.");
        }
      }
    }

    void loadWorkspace();
    return () => {
      isCurrent = false;
    };
  }, []);

  function handleExperimentUpdated(experiment: ExperimentRecord) {
    setExperiments((current) => {
      const existingIndex = current.findIndex((item) => item.exp_id === experiment.exp_id);
      if (existingIndex === -1) {
        return [...current, experiment];
      }
      return current.map((item) => (item.exp_id === experiment.exp_id ? experiment : item));
    });
  }

  const baseConfig = configs.find((config) => config.name === "base");

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top">VISHUSTRA<span>•</span></a>
        <p>Transformer intelligence research laboratory</p>
        <span className={health?.status === "ok" ? "connection online" : "connection"}>
          <i aria-hidden="true" />
          {health?.status === "ok" ? `API ${health.version}` : "Connecting"}
        </span>
      </header>

      <div className="hero" id="top">
        <p className="eyebrow">Where attention becomes visible</p>
        <h1>Observe the model.<br /><em>Challenge the mechanism.</em></h1>
        <p className="hero-copy">
          A working research environment for inspecting Transformer calculations and preserving every experiment.
        </p>
      </div>

      {error ? <p className="api-error" role="alert">API connection failed: {error}</p> : null}

      <section className="metric-grid" aria-label="Laboratory summary">
        <MetricCard label="Paper configurations" value={String(configs.length).padStart(2, "0")} detail="base and big references" />
        <MetricCard label="Base model" value={baseConfig ? `${baseConfig.num_heads}H` : "—"} detail={baseConfig ? `${baseConfig.d_model} dimensions` : "loading reference"} />
        <MetricCard label="Experiments" value={String(experiments.length).padStart(2, "0")} detail="tracked in the registry" />
      </section>

      <div className="lab-grid">
        <AttentionLaboratory />
        <ExperimentWorkspace experiments={experiments} onExperimentUpdated={handleExperimentUpdated} />
      </div>
    </main>
  );
}

export default App;
