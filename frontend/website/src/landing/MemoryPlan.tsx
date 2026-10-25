import { useId, useState } from 'react';
import './memory-plan.css';

const plans = [
  {
    label: 'GPU',
    title: 'GPU-resident layers',
    description: 'Weights stay in GPU memory. The local GPU is filled first; compatible peers are considered when the model exceeds local GPU memory.',
    gpuStart: 1,
    gpuLabel: 'All layer groups',
    ramGroups: [] as number[],
    ramLabel: 'No CPU layers assigned',
    gpuNote: 'Layers execute on the GPUs that hold their weights.',
    ramNote: 'The host still needs memory to run the process and prepare inputs.',
    diskNote: 'Weights are loaded before execution. No ongoing weight paging is shown.',
    allocation: 'All twelve illustrative layer groups are resident in GPU memory, with separate room for KV cache and compute buffers. Host runtime memory is reserved separately.',
  },
  {
    label: 'GPU + RAM',
    title: 'CPU overflow',
    description: 'Earlier layers can execute on the host CPU from RAM, with later layers on GPUs. The planner compares this split with sending work to a compatible peer.',
    gpuStart: 5,
    gpuLabel: 'Later layer groups',
    ramGroups: [1, 2, 3, 4],
    ramLabel: 'Earlier layer groups',
    gpuNote: 'The later layers remain GPU-resident.',
    ramNote: 'The CPU executes these layers directly from host RAM.',
    diskNote: 'Both GPU and CPU weights remain resident after loading.',
    allocation: 'Illustrative groups one through four execute on the CPU from host RAM; groups five through twelve execute on GPUs. Each memory pool also reserves runtime and cache space.',
  },
  {
    label: 'Disk-backed',
    title: 'Disk-backed paging',
    description: 'In this capacity-mode example, the CPU portion exceeds available RAM. Weight pages are read from the model file as needed; storage reads slow generation.',
    gpuStart: 7,
    gpuLabel: 'GPU-resident layer groups',
    ramGroups: [1, 2],
    ramLabel: 'Resident weight pages · example window',
    gpuNote: 'These weights remain on the GPU throughout execution.',
    ramNote: 'Only a working set of CPU weight pages is resident at a time.',
    diskNote: 'Backing for CPU groups 01–06. Pages must reach RAM before the CPU uses them.',
    allocation: 'Illustrative groups seven through twelve stay on GPUs. CPU groups one through six are disk-backed, with a small example page window in host RAM. KV cache and runtime buffers still require resident memory.',
  },
] as const;

function MemoryHardware({ kind }: { kind: 'gpu' | 'ram' | 'disk' }) {
  return (
    <svg className="hc-memory-hardware" viewBox="0 0 104 62" fill="none" aria-hidden="true">
      {kind === 'gpu' ? <>
        <path d="M10 10h84v39H10z" fill="#d4e4f6" />
        <path d="M6 5v51m0-5h12m-8-41h84v39H10z" />
        <rect x="37" y="17" width="29" height="25" fill="#f4f6f8" />
        <path d="M43 22h17v15H43zM25 49v7h46v-7M31 51v5m6-5v5m6-5v5m6-5v5m6-5v5m6-5v5m6-5v5" />
        {[18, 77].map(x => <g key={x}><rect x={x} y="17" width="10" height="9" fill="#a9c7e9" /><rect x={x} y="32" width="10" height="9" fill="#a9c7e9" /></g>)}
        <path d="M32 20h5m-5 6h5m-5 7h5m29-13h6m-6 6h6m-6 7h6" />
      </> : kind === 'ram' ? <>
        <path d="M6 17h92v29H6z" fill="#dce3e9" />
        {[13, 34, 55, 76].map(x => <g key={x}><rect x={x} y="23" width="14" height="15" fill="#f4f6f8" /><path d={`M${x + 3} 20v3m4-3v3m4-3v3M${x + 3} 38v3m4-3v3m4-3v3`} /></g>)}
        <path d="M12 46v8h35v-5h7v5h38v-8m-74 2v6m6-6v6m6-6v6m6-6v6m6-6v6m18-6v6m6-6v6m6-6v6m6-6v6m6-6v6" />
      </> : <>
        <path d="M21 7h62v47H21z" fill="#e5e8eb" />
        <rect x="29" y="14" width="46" height="31" fill="#f4f6f8" />
        <path d="M36 23h32m-32 7h22m-22 7h27M31 54v4h14v-4m6 0v4h20v-4" />
        <circle cx="25" cy="11" r="1" fill="#000" /><circle cx="79" cy="50" r="1" fill="#000" />
      </>}
    </svg>
  );
}

function WeightGroups({ groups, paged = false }: { groups: readonly number[]; paged?: boolean }) {
  return (
    <div className="hc-memory-groups" data-paged={paged}>
      {groups.map(group => <span className="hc-memory-group" key={group}>{String(group).padStart(2, '0')}</span>)}
    </div>
  );
}

export function MemoryPlan() {
  const [selected, setSelected] = useState(0);
  const figureId = useId();
  const plan = plans[selected];
  const paged = selected === 2;
  const gpuGroups = Array.from({ length: 13 - plan.gpuStart }, (_, index) => plan.gpuStart + index);

  return (
    <div className="hc-memory-plan">
      <div className="hc-memory-controls" role="group" aria-label="Explore memory placement">
        {plans.map((item, index) => (
          <button type="button" key={item.label} aria-pressed={selected === index} aria-controls={figureId} onClick={() => setSelected(index)}>
            {item.label}
          </button>
        ))}
      </div>
      <figure className="hc-memory-figure" id={figureId} aria-label={plan.allocation}>
        <figcaption className="hc-memory-caption">
          <span>Memory placement</span>
          <span>12 illustrative layer groups · 01–12</span>
        </figcaption>

        <div className="hc-memory-row" data-device="gpu">
          <div className="hc-memory-device"><MemoryHardware kind="gpu" /><h4>GPU VRAM</h4><span>GPU execution</span></div>
          <div className="hc-memory-allocation">
            <div className="hc-memory-weights"><span className="hc-memory-area-label">{plan.gpuLabel}</span><WeightGroups groups={gpuGroups} /><span className="hc-memory-area-detail">Model weights</span></div>
            <div className="hc-memory-reserved"><span className="hc-memory-area-label">Reserved · resident</span><div className="hc-memory-reserve-blocks"><span>KV cache</span><span>Compute<br />buffers</span></div></div>
          </div>
          <p className="hc-memory-row-note">{plan.gpuNote}</p>
        </div>

        <div className="hc-memory-row" data-device="ram">
          <div className="hc-memory-device"><MemoryHardware kind="ram" /><h4>Host RAM</h4><span>CPU execution</span></div>
          <div className="hc-memory-allocation">
            <div className="hc-memory-weights"><span className="hc-memory-area-label">{plan.ramLabel}</span>{plan.ramGroups.length ? <WeightGroups groups={plan.ramGroups} paged={paged} /> : <div className="hc-memory-unassigned">Available for other host work</div>}<span className="hc-memory-area-detail">{paged ? 'Pages enter and leave this working set' : plan.ramGroups.length ? 'Model weights' : 'No model weights in this example'}</span></div>
            <div className="hc-memory-reserved"><span className="hc-memory-area-label">Reserved · resident</span><div className="hc-memory-reserve-blocks"><span>Host<br />runtime</span><span>{selected === 0 ? <>Input<br />buffers</> : <>KV cache<br />+ buffers</>}</span></div></div>
          </div>
          <p className="hc-memory-row-note">{plan.ramNote}</p>
        </div>

        <div className="hc-memory-row" data-device="disk">
          <div className="hc-memory-device"><MemoryHardware kind="disk" /><h4>Storage</h4><span>Weight backing</span></div>
          <div className="hc-memory-allocation hc-memory-disk-allocation">
            <div className="hc-memory-weights"><span className="hc-memory-area-label">{paged ? 'Disk-backed CPU weight groups' : 'Model file'}</span>{paged ? <WeightGroups groups={[1, 2, 3, 4, 5, 6]} paged /> : <div className="hc-memory-source-file"><span className="hc-memory-file-icon" aria-hidden="true" /><span>Saved model weights</span><span>Read at load time</span></div>}<span className="hc-memory-area-detail">{paged ? 'Backing store · not execution memory' : 'No weight paging during this example'}</span></div>
            <div className="hc-memory-disk-path"><span className="hc-memory-area-label">{paged ? 'On demand' : 'At startup'}</span><span>Disk <span aria-hidden="true">→</span> RAM</span><span>{paged ? 'Page reads during inference' : 'Load into the assigned devices'}</span></div>
          </div>
          <p className="hc-memory-row-note">{plan.diskNote}</p>
        </div>

        <div className="hc-memory-key"><span><i data-kind="weights" />Model weights</span><span><i data-kind="reserved" />Resident cache / runtime</span><span><i data-kind="pages" />Disk-backed weight pages</span></div>
      </figure>
      <div className="hc-memory-explanation" aria-live="polite" aria-atomic="true"><h3>{plan.title}</h3><p>{plan.description}</p></div>
      <p className="hc-diagram-note">Group numbers and block widths are illustrative, not a model specification or memory measurement. Cache placement depends on the backend. Disk backing does not remove the need for resident KV cache and runtime memory.</p>
    </div>
  );
}
