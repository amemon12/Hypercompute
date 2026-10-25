import { useState } from 'react';
import { InferenceTimeline } from './InferenceTimeline';
import './execution-diagram.css';

const devices = [
  { name: 'Laptop A', range: '00–15', spokenRange: '00 to 15', desktop: false },
  { name: 'Desktop B', range: '16–31', spokenRange: '16 to 31', desktop: true },
  { name: 'Laptop C', range: '32–47', spokenRange: '32 to 47', desktop: false },
] as const;

function DeviceIllustration({ desktop }: { desktop: boolean }) {
  return (
    <svg viewBox="0 0 120 94" aria-hidden="true" focusable="false">
      {desktop ? (
        <>
          <rect className="hd-case" x="34" y="2" width="52" height="82" rx="3" />
          <path d="M45 9v68M55 13h22m-22 6h22m-22 6h22" fill="none" />
          <rect className="hd-display" x="55" y="35" width="22" height="27" />
          <path d="M72 75h5" fill="none" />
        </>
      ) : (
        <>
          <rect className="hd-case" x="15" y="21" width="90" height="62" rx="3" />
          <rect className="hd-display" x="19" y="25" width="82" height="54" rx="1" />
          <path className="hd-case" d="M5 84h110v3c0 4-6 6-11 6H16C11 93 5 91 5 87z" />
          <path d="M47 84h26l-3 3H50z" fill="none" />
        </>
      )}
    </svg>
  );
}

export function ExecutionDiagram() {
  const [activeDevice, setActiveDevice] = useState<number | null>(null);
  const toggleDevice = (index: number) => setActiveDevice(current => current === index ? null : index);

  return (
    <div className="hc-distribution" data-selected={activeDevice ?? 'none'}>
      <div className="hd-row hd-devices-row">
        <div className="hd-copy">
          <h2>Distributed across your devices</h2>
          <p>Each computer holds part of the model. Together, they can run models that don’t fit on one machine.</p>
        </div>
        <div className="hd-devices" aria-label="Three connected computers">
          <span className="hd-link hd-link-first" aria-hidden="true" />
          <span className="hd-link hd-link-second" aria-hidden="true" />
          {devices.map((device, index) => (
            <button
              key={device.name}
              type="button"
              className="hd-device"
              data-device={index}
              aria-pressed={activeDevice === index}
              aria-label={`Highlight ${device.name}, layers ${device.spokenRange}`}
              onClick={() => toggleDevice(index)}
            >
              <span className="hd-device-art"><DeviceIllustration desktop={device.desktop} /></span>
              <span className="hd-device-name">{device.name}</span>
              <span className="hd-device-range">Layers {device.range}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="hd-row hd-split-row">
        <div className="hd-copy">
          <h3>Model split view</h3>
          <p>This example splits 48 layers across three GPUs. The split depends on available memory.</p>
        </div>
        <figure className="hd-split" aria-label="A 48-layer model divided into three groups of 16 layers">
          <figcaption className="hd-layer-total"><span>48 layers</span></figcaption>
          <div className="hd-layer-groups">
            {devices.map((device, index) => (
              <div className="hd-layer-allocation" key={device.name} data-device={index} data-active={activeDevice === index}>
                <div className="hd-layer-blocks" aria-hidden="true">
                  {Array.from({ length: 16 }, (_, layer) => <i key={layer} />)}
                </div>
                <div className="hd-layer-bracket" aria-hidden="true" />
                <span className="hd-allocation-name">{device.name}</span>
                <span className="hd-allocation-count">16 layers</span>
              </div>
            ))}
          </div>
        </figure>
      </div>

      <div className="hd-row hd-timeline-row">
        <div className="hd-copy">
          <h3>Inference timeline</h3>
          <p>For each token, the GPUs run their layers in sequence.</p>
        </div>
        <InferenceTimeline activeDevice={activeDevice} />
      </div>
    </div>
  );
}
