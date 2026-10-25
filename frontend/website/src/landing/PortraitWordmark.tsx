import './portrait-wordmark.css';

// Panel boundaries follow the six letter pairs in the static Anton glyph mask.
const portraits = [
  { person: 'huang', width: 16.056584 },
  { person: 'akeley', width: 17.149301 },
  { person: 'hanrahan', width: 14.580398 },
  { person: 'su', width: 21.634556 },
  { person: 'malachowsky', width: 17.331024 },
  { person: 'dally', width: 13.248137 },
];

export function PortraitWordmark() {
  return (
    <div className="hc-portrait-wordmark" role="img" aria-label="Hypercompute">
      {portraits.map(({ person, width }) => (
        <span
          key={person}
          className="hc-portrait-panel"
          data-person={person}
          style={{ width: `${width}%` }}
          aria-hidden="true"
        />
      ))}
    </div>
  );
}
