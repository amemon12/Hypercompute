/** The hardware cutout and print accents share the page's paper canvas. */
export function HeroGpuArt() {
  return (
    <figure className="hc-hero-art">
      <svg className="hc-hero-print" viewBox="0 0 1200 360" aria-hidden="true" focusable="false">
        <path className="hc-hero-print-block" d="M760 30H1130V95C1036 95 1036 217 1130 217V300H660V170Z" />
        <g className="hc-hero-print-marks">
          <path d="M75 40V70M60 55H90M1130 15V60M1110 40H1150" />
          <path d="M65 262C-4 288 25 313 120 263L342 151M862 313C1048 193 1159 115 1152 163C1146 200 1091 263 1086 326" />
        </g>
      </svg>
      <img
        className="hc-board-image"
        src="/brand/gpu-hardware-cutout.png"
        alt="Hand-drawn GPU with copper circuitry, two exposed chips, and a cooling fan"
        width="1920"
        height="819"
        decoding="async"
      />
    </figure>
  );
}
