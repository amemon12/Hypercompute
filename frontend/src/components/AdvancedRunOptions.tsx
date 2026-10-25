import type { EnginesStatus, ModelArtifact, Objective } from '../api/types';
import { ENGINE_CHOICES, ENGINE_HINTS, ENGINE_LABELS, OBJECTIVE_HINTS, OBJECTIVE_LABELS, engineChoiceState } from '../lib/engines';
import { advancedSummary, type EngineChoice, type RunOptions } from '../lib/runOptions';

const OBJECTIVES: Objective[] = ['interactive', 'throughput'];

interface AdvancedRunOptionsProps {
  options: RunOptions;
  /** The engine actually sent: a forced engine this model cannot use reads as automatic. */
  engine: EngineChoice;
  engines: EnginesStatus | undefined;
  artifact: ModelArtifact | null;
  onChange: (next: RunOptions) => void;
}

interface ToggleProps {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (on: boolean) => void;
}

function Toggle({ label, hint, checked, onChange }: ToggleProps) {
  return (
    <label className="choice">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>
        {label}
        <span className="muted small block">{hint}</span>
      </span>
    </label>
  );
}

export function AdvancedRunOptions({ options, engine, engines, artifact, onChange }: AdvancedRunOptionsProps) {
  const set = (patch: Partial<RunOptions>): void => onChange({ ...options, ...patch });
  const changed = advancedSummary({ ...options, engine });
  return (
    <details className="advanced">
      <summary>
        Advanced: engine, objective and limits
        {changed.length > 0 && <span className="muted small"> · {changed.join(' · ')}</span>}
      </summary>

      <fieldset className="choice-group">
        <legend>Engine</legend>
        <label className="choice">
          <input type="radio" name="engine" checked={engine === 'auto'} onChange={() => set({ engine: 'auto' })} />
          <span>
            Automatic
            <span className="muted small block">Picks the engine from the model&apos;s format and the machines you chose.</span>
          </span>
        </label>
        {ENGINE_CHOICES.map((id) => {
          const state = engineChoiceState(id, engines, artifact);
          return (
            <label key={id} className={`choice ${state.enabled ? '' : 'choice-off'}`}>
              <input
                type="radio"
                name="engine"
                disabled={!state.enabled}
                checked={engine === id}
                onChange={() => set({ engine: id })}
              />
              <span>
                {ENGINE_LABELS[id]}
                <span className="muted small block">{state.reason ? `Unavailable: ${state.reason}` : ENGINE_HINTS[id]}</span>
              </span>
            </label>
          );
        })}
      </fieldset>

      <fieldset className="choice-group">
        <legend>Who will use it</legend>
        {OBJECTIVES.map((o) => (
          <label key={o} className="choice">
            <input type="radio" name="objective" checked={options.objective === o} onChange={() => set({ objective: o })} />
            <span>
              {OBJECTIVE_LABELS[o]}
              <span className="muted small block">{OBJECTIVE_HINTS[o]}</span>
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset className="choice-group">
        <legend>Limits</legend>
        <Toggle
          label="Use a measured profile when one exists"
          hint="Reuses the fastest setup a tune measured for this model and these machines."
          checked={options.useProfiles}
          onChange={(on) => set({ useProfiles: on })}
        />
        <Toggle
          label="Allow this PC's RAM"
          hint="When the GPUs cannot hold the whole model, the first layers run on this PC's CPU. Slower than all-GPU."
          checked={options.allowRamTier}
          onChange={(on) => set({ allowRamTier: on })}
        />
        <Toggle
          label="Allow capacity mode"
          hint="As a last resort, read layers from disk while it runs. Expect seconds or more per token."
          checked={options.allowCapacityMode}
          onChange={(on) => set({ allowCapacityMode: on })}
        />
      </fieldset>
    </details>
  );
}
