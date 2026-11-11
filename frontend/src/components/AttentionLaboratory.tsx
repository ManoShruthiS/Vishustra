import { type FormEvent, useState } from "react";

import { analyzeAttention } from "../api";
import type { AttentionResult } from "../types";

const DEFAULT_TOKENS = "the, scientist, studied, the, paper";

function AttentionMatrix({ result }: { result: AttentionResult }) {
  return (
    <div className="matrix-scroll" role="region" aria-label="Attention weight matrix" tabIndex={0}>
      <table className="attention-matrix">
        <thead>
          <tr>
            <th scope="col">query ↓ / key →</th>
            {result.tokens.map((token, index) => (
              <th scope="col" key={`${token}-${index}`}>
                {token}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.weights.map((row, rowIndex) => (
            <tr key={`${result.tokens[rowIndex]}-${rowIndex}`}>
              <th scope="row">{result.tokens[rowIndex]}</th>
              {row.map((weight, columnIndex) => (
                <td
                  aria-label={`${result.tokens[rowIndex]} attends to ${result.tokens[columnIndex]}: ${weight.toFixed(3)}`}
                  key={`${rowIndex}-${columnIndex}`}
                  style={{ backgroundColor: `rgba(83, 255, 160, ${0.08 + weight * 0.82})` }}
                >
                  {weight.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AttentionLaboratory() {
  const [tokensText, setTokensText] = useState(DEFAULT_TOKENS);
  const [dimension, setDimension] = useState(8);
  const [seed, setSeed] = useState(7);
  const [result, setResult] = useState<AttentionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const tokens = tokensText
      .split(",")
      .map((token) => token.trim())
      .filter(Boolean);

    if (tokens.length === 0) {
      setError("Add at least one comma-separated token.");
      return;
    }

    setError(null);
    setIsAnalyzing(true);
    try {
      setResult(await analyzeAttention(tokens, dimension, seed));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The attention calculation could not be completed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <section className="laboratory-panel" aria-labelledby="attention-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Laboratory 01</p>
          <h2 id="attention-title">Attention Observatory</h2>
        </div>
        <p>Inspect the complete scaled dot-product attention calculation.</p>
      </div>

      <form className="attention-controls" onSubmit={handleSubmit}>
        <label className="token-input">
          <span>Tokens</span>
          <input
            aria-describedby="tokens-help"
            onChange={(event) => setTokensText(event.target.value)}
            value={tokensText}
          />
          <small id="tokens-help">Separate words with commas.</small>
        </label>
        <label>
          <span>Dimension dₖ</span>
          <input
            max="64"
            min="1"
            onChange={(event) => setDimension(Number(event.target.value))}
            type="number"
            value={dimension}
          />
        </label>
        <label>
          <span>Seed</span>
          <input
            min="0"
            onChange={(event) => setSeed(Number(event.target.value))}
            type="number"
            value={seed}
          />
        </label>
        <button className="primary-button" disabled={isAnalyzing} type="submit">
          {isAnalyzing ? "Calculating…" : "Run attention"}
        </button>
      </form>

      {error ? <p className="form-error" role="alert">{error}</p> : null}
      {result ? (
        <div className="attention-result">
          <div className="calculation-summary">
            <span>Attention(Q, K, V)</span>
            <strong>softmax(QKᵀ / √{result.d_k})V</strong>
            <small>Scaling factor: {result.scaler.toFixed(4)}</small>
          </div>
          <AttentionMatrix result={result} />
        </div>
      ) : (
        <p className="empty-state">Run the calculation to reveal how every token distributes its attention.</p>
      )}
    </section>
  );
}
