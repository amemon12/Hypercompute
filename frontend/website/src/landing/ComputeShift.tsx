import './compute-shift.css';

const layerTiles = [0, 1, 2, 3, 4, 5];

export function ComputeShift() {
  return (
    <div
      className="compute-shift"
      role="img"
      aria-label="Concept illustration of one model’s layers divided across a laptop and other connected computers."
    >
      <div className="compute-shift__model">
        <svg className="compute-shift__model-art" viewBox="0 0 240 210" aria-hidden="true" focusable="false">
          <g className="compute-shift__bank">
            {layerTiles.map(tile => (
              <rect
                key={tile}
                x={30 + tile * 31}
                y="38"
                width="23"
                height="136"
                fill="currentColor"
                opacity={tile < 2 ? 0.3 : tile < 4 ? 0.58 : 1}
              />
            ))}
          </g>
        </svg>
        <span className="compute-shift__label">One model</span>
      </div>

      <svg className="compute-shift__handoff" viewBox="0 0 48 24" aria-hidden="true" focusable="false">
        <path d="M3 12h39m-8-8 8 8-8 8" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>

      <div className="compute-shift__computers">
        <svg className="compute-shift__computer-art" viewBox="0 0 620 280" aria-hidden="true" focusable="false">
          <g className="compute-shift__peer compute-shift__peer--desktop">
            <path className="compute-shift__outline" d="M29 105h73v122H29zM42 119h47M42 126h47M79 212h10" />
            <g className="compute-shift__allocation compute-shift__allocation--first" opacity="0.3">
              <rect x="43" y="143" width="16" height="50" fill="currentColor" />
              <rect x="66" y="143" width="16" height="50" fill="currentColor" />
            </g>
          </g>

          <g className="compute-shift__connection" fill="currentColor" opacity="0.4">
            <circle cx="125" cy="175" r="2" />
            <circle cx="137" cy="175" r="2" />
            <circle cx="149" cy="175" r="2" />
          </g>

          <g className="compute-shift__laptop">
            <path className="compute-shift__outline" d="M196 45h208v139H196zM177 190h246v7l-14 9H191l-14-9zM278 190h44l-5 5h-34z" />
            <g className="compute-shift__allocation compute-shift__allocation--second" opacity="0.58">
              <rect x="259" y="76" width="32" height="77" fill="currentColor" />
              <rect x="307" y="76" width="32" height="77" fill="currentColor" />
            </g>
          </g>

          <g className="compute-shift__connection" fill="currentColor" opacity="0.4">
            <circle cx="447" cy="175" r="2" />
            <circle cx="459" cy="175" r="2" />
            <circle cx="471" cy="175" r="2" />
          </g>

          <g className="compute-shift__peer compute-shift__peer--laptop">
            <path className="compute-shift__outline" d="M500 111h99v75h-99zM490 192h119v5l-7 6H497l-7-6zM539 192h22" />
            <g className="compute-shift__allocation compute-shift__allocation--third">
              <rect x="529" y="129" width="14" height="38" fill="currentColor" />
              <rect x="554" y="129" width="14" height="38" fill="currentColor" />
            </g>
          </g>
        </svg>
        <span className="compute-shift__label">Across your computers</span>
      </div>
    </div>
  );
}
