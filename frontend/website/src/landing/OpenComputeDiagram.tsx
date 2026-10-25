import { useId } from 'react';
import './open-compute-diagram.css';

const CENTER_X = 220;
const HALF_WIDTH = 204;

type PlaneProps = {
  y: number;
  depth: number;
  face: string;
  edge: string;
  side: string;
  ink: string;
  hatched?: boolean;
};

/** A square board projected into the same isometric space as its components. */
function Plane({ y, depth, face, edge, side, ink, hatched }: PlaneProps) {
  const left = CENTER_X - HALF_WIDTH;
  const right = CENTER_X + HALF_WIDTH;
  const front = y + HALF_WIDTH / 2;
  const back = y - HALF_WIDTH / 2;

  return (
    <g stroke={ink} strokeWidth="1.3" strokeLinejoin="round">
      <path d={`M${left} ${y} L${CENTER_X} ${front} L${right} ${y} V${y + depth} L${CENTER_X} ${front + depth} L${left} ${y + depth} Z`} fill={edge} />
      <path d={`M${CENTER_X} ${front} L${right} ${y} V${y + depth} L${CENTER_X} ${front + depth} Z`} fill={side} />
      {hatched && (
        <g strokeWidth=".45" opacity=".58">
          {Array.from({ length: 65 }, (_, index) => {
            const offset = 3 + index * 3.1;
            return (
              <g key={index}>
                <path d={`M${left + offset} ${y + offset / 2 + 2} v${depth - 4}`} />
                <path d={`M${CENTER_X + offset} ${front - offset / 2 + 2} v${depth - 4}`} />
              </g>
            );
          })}
        </g>
      )}
      <path d={`M${left} ${y} L${CENTER_X} ${back} L${right} ${y} L${CENTER_X} ${front} Z`} fill={face} />
    </g>
  );
}

/** Repeated memory packages and their etched contacts on the face of a board. */
function Packages({ blue = false }: { blue?: boolean }) {
  return (
    <g fill={blue ? '#9BB7C6' : '#E7E3D8'} stroke={blue ? '#547787' : '#89877C'} strokeWidth=".65">
      {[0, 90, 180, 270].map((angle) => (
        <g key={angle} transform={`rotate(${angle})`}>
          <rect x="-28" y="-91" width="56" height="15" rx="1.2" />
          {Array.from({ length: 13 }, (_, index) => (
            <path key={index} d={`M${-24 + index * 4} -95v4 M${-24 + index * 4} -76v4`} opacity=".8" />
          ))}
          <path d="M-23-86h46M-23-82h46" strokeWidth=".4" opacity=".6" />
        </g>
      ))}
    </g>
  );
}

function Fixings({ blue = false }: { blue?: boolean }) {
  return (
    <g fill={blue ? '#D5E1E5' : '#F3EFE4'} stroke={blue ? '#5E7B8C' : '#706F65'} strokeWidth=".8">
      {[-89, 89].flatMap((x) => [-89, 89].map((y) => (
        <g key={`${x}-${y}`}>
          <circle cx={x} cy={y} r="3.6" />
          <path d={`M${x - 1.6} ${y} h3.2`} strokeWidth=".55" />
        </g>
      )))}
    </g>
  );
}

function BoardTraces({ blue = false }: { blue?: boolean }) {
  return (
    <g stroke={blue ? '#64899C' : '#A39E90'} strokeWidth=".6" opacity={blue ? '.8' : '.7'} fill="none">
      {[0, 90, 180, 270].map((angle) => (
        <g key={angle} transform={`rotate(${angle})`}>
          <path d="M-50-28h-13v-35h-20v-11M-50-17h-25v-31h-16M-50-6h-33v-13h-10M-50 8h-29v25h-12M-50 20h-14v39h-22v14" />
          <path d="M-70-59h-10M-84-42h-10M-88 10h-7M-80 49h-14" />
          {[-60, -38, 36, 57].map((y) => <circle key={y} cx="-91" cy={y} r="1" />)}
        </g>
      ))}
    </g>
  );
}

export function OpenComputeDiagram() {
  const id = useId().replace(/:/g, '');
  const contacts = `${id}-contacts`;
  const fineContacts = `${id}-fine-contacts`;
  const dieCells = `${id}-die-cells`;
  const hatch = `${id}-etch`;

  return (
    <figure className="hc-open-compute" aria-label="Model partition, coordinated execution, and GPU memory">
      <svg className="hc-open-compute-art" viewBox="0 0 440 550" fill="none" role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>An exploded view of distributed computing</title>
        <desc id={`${id}-description`}>Three separated circuit boards represent a model partition, coordinated execution, and GPU memory. The middle execution board is pale blue.</desc>
        <defs>
          <pattern id={contacts} width="5" height="5" patternUnits="userSpaceOnUse">
            <circle cx="2.5" cy="2.5" r=".72" fill="#646153" opacity=".64" />
            <path d="M1.8 1.8h1.4" stroke="#F5F0E6" strokeWidth=".5" />
          </pattern>
          <pattern id={fineContacts} width="3.5" height="3.5" patternUnits="userSpaceOnUse">
            <circle cx="1.75" cy="1.75" r=".45" fill="#8A8678" opacity=".55" />
          </pattern>
          <pattern id={dieCells} width="5" height="5" patternUnits="userSpaceOnUse">
            <path d="M.8.8h3.4v3.4H.8zM2.5.8v3.4M.8 2.5h3.4" stroke="#365A6C" strokeWidth=".28" opacity=".65" />
          </pattern>
          <pattern id={hatch} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(-12)">
            <path d="M0 1h4" stroke="#8C897E" strokeWidth=".35" opacity=".44" />
          </pattern>
        </defs>

        <g className="hc-open-compute-guides" stroke="#8A867A" strokeWidth=".75" strokeDasharray="3 5" opacity=".55">
          <path d="M16 144v266M424 144v266M220 247v155" />
        </g>

        <g data-layer="memory">
          <Plane y={410} depth={24} face="#EEEADF" edge="#DAD6C9" side="#CDCABC" ink="#414137" hatched />
          <g transform="matrix(1 .5 -1 .5 220 410)" strokeLinejoin="round">
            <rect x="-96" y="-96" width="192" height="192" stroke="#A39D8D" strokeWidth=".6" />
            <BoardTraces />
            <rect x="-75" y="-75" width="150" height="150" rx="1.5" fill="#E3DED0" stroke="#706D5F" strokeWidth="1" />
            <rect x="-67" y="-67" width="134" height="134" fill="#F1ECDF" stroke="#908B7B" strokeWidth=".65" />
            <rect x="-62" y="-62" width="124" height="124" fill={`url(#${contacts})`} stroke="#AEA794" strokeWidth=".5" />
            <path d="M-79-61v122M79-61v122M-61-79H61M-61 79H61" stroke="#848071" strokeWidth="1.4" />
            <Packages />
            <Fixings />
          </g>
          <g stroke="#666353" strokeWidth=".7" fill="#C8C4B7">
            <path d="m43 425 13 6.5v21l-13-6.5zM179 493l13 6.5v21l-13-6.5zM248 499l13-6.5v21l-13 6.5zM385 430.5l13-6.5v21l-13 6.5z" />
            <path d="m45 429 9 4.5m127 65 9 4.5m60 1 9-4.5m128-65 9-4.5" />
          </g>
        </g>

        <g data-layer="execution">
          <Plane y={270} depth={12} face="#C9DCE5" edge="#9AB8CC" side="#7E9EB6" ink="#3D5B6C" />
          <g transform="matrix(1 .5 -1 .5 220 270)" strokeLinejoin="round">
            <rect x="-96" y="-96" width="192" height="192" stroke="#88A5B4" strokeWidth=".65" />
            <BoardTraces blue />
            <Packages blue />
            <rect x="-54" y="-54" width="108" height="108" rx="1.5" fill="#A9C3D1" stroke="#486574" strokeWidth="1" />
            <rect x="-46" y="-46" width="92" height="92" fill="#7298AC" stroke="#344F5F" strokeWidth="1" />
            <rect x="-42" y="-42" width="84" height="84" fill={`url(#${dieCells})`} />
            <g stroke="#345567" strokeWidth=".65" opacity=".78">
              <path d="M-14-42v84M14-42v84M-42-14h84M-42 14h84" />
              <path d="M-42-38h84M-42 38h84M-38-42v84M38-42v84" strokeWidth=".4" />
            </g>
            <g fill="#97B6C6" stroke="#567C8F" strokeWidth=".5">
              {[-62, 58].flatMap((x) => [-37, -20, -3, 14, 31].map((y) => <rect key={`${x}-${y}`} x={x} y={y} width="4" height="7" />))}
            </g>
            <Fixings blue />
          </g>
          <path d="m16 274 204 102 204-102M16 278l204 102 204-102" stroke="#54788D" strokeWidth=".45" opacity=".65" />
        </g>

        <g data-layer="model">
          <Plane y={130} depth={14} face="#F4F0E5" edge="#E5E0D4" side="#D5D1C5" ink="#414137" />
          <path d="m16 134 204 102 204-102M16 138l204 102 204-102" stroke="#908B7D" strokeWidth=".55" />
          <g transform="matrix(1 .5 -1 .5 220 130)" strokeLinejoin="round">
            <rect x="-96" y="-96" width="192" height="192" stroke="#B5AE9C" strokeWidth=".6" />
            <BoardTraces />
            <rect x="-67" y="-67" width="134" height="134" fill="#F2EEE2" stroke="#8D887A" strokeWidth=".8" />
            <rect x="-61" y="-61" width="122" height="122" fill={`url(#${fineContacts})`} stroke="#B0A997" strokeWidth=".5" />
            <Packages />
            <g fill={`url(#${hatch})`}>
              <rect x="-28" y="-91" width="56" height="15" />
              <rect x="-28" y="76" width="56" height="15" />
              <rect x="-91" y="-28" width="15" height="56" />
              <rect x="76" y="-28" width="15" height="56" />
            </g>
            <Fixings />
          </g>
        </g>

      </svg>
    </figure>
  );
}
