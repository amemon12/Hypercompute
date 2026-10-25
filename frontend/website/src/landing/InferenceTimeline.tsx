import { useId } from 'react';

type InferenceTimelineProps = {
  activeDevice: number | null;
};

const devices = [
  { name: 'Laptop A', color: '#98ad92', stroke: '#5d7657', x: 128, y: 30 },
  { name: 'Desktop B', color: '#d69876', stroke: '#a46744', x: 314, y: 76 },
  { name: 'Laptop C', color: '#ccc9bd', stroke: '#8d897a', x: 500, y: 122 },
];

/** Illustrates execution order only; bar lengths do not represent measured latency. */
export function InferenceTimeline({ activeDevice }: InferenceTimelineProps) {
  const id = useId();
  const titleId = `${id}-title`;
  const descriptionId = `${id}-description`;
  const arrowId = `${id}-arrow`;

  return (
    <svg
      className="hc-inference-timeline"
      viewBox="0 0 720 170"
      width="720"
      height="170"
      role="img"
      aria-labelledby={`${titleId} ${descriptionId}`}
      style={{ display: 'block', width: '100%', height: 'auto', overflow: 'visible' }}
    >
      <title id={titleId}>Illustrative inference sequence across three computers</title>
      <desc id={descriptionId}>
        Laptop A runs layers 00 through 15, then passes the result to Desktop B
        for layers 16 through 31. Laptop C finishes layers 32 through 47.
        The timeline shows execution order, not measured timing.
      </desc>

      <defs>
        <marker id={arrowId} viewBox="0 0 7 7" refX="6" refY="3.5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M1 1L5.5 3.5L1 6" fill="none" stroke="#68675e" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>

      <g fill="none" stroke="#bdb8aa" strokeWidth="0.8">
        {[30, 76, 122].map(y => <path key={y} d={`M112 ${y}H706`} opacity="0.38" />)}
        {[124, 310, 496, 682].map(x => <path key={x} d={`M${x} 4V143`} strokeDasharray="2 3" opacity="0.55" />)}
      </g>

      <g fontFamily="Arial, Helvetica, sans-serif" fontSize="18" fill="#262621">
        {devices.map((device, index) => (
          <g key={device.name}>
            <text x="94" y={device.y} textAnchor="end" dominantBaseline="central">{device.name}</text>
            <rect
              x={device.x}
              y={device.y - 12}
              width="136"
              height="24"
              rx="2"
              fill={device.color}
              fillOpacity={activeDevice === null || activeDevice === index ? 1 : 0.62}
              stroke={activeDevice === index ? device.stroke : 'none'}
              strokeWidth="1.6"
            />
          </g>
        ))}
      </g>

      <g fill="none" stroke="#68675e" strokeWidth="1" strokeLinecap="round" strokeDasharray="2 3" markerEnd={`url(#${arrowId})`}>
        <path d="M266 30H290Q298 30 298 38V68Q298 76 306 76H312" />
        <path d="M452 76H476Q484 76 484 84V114Q484 122 492 122H498" />
      </g>

      <g fill="#5b5a52" fontFamily="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" fontSize="16" textAnchor="middle">
        {[124, 310, 496, 682].map((x, index) => <text key={x} x={x} y="161">t{index}</text>)}
      </g>
    </svg>
  );
}
