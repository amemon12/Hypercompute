export type PrintKind = 'steps' | 'orbit' | 'plates' | 'connections' | 'corner' | 'registration';
export function PrintArt({ kind, className = '' }: { kind: PrintKind; className?: string }) {
  return <img className={`hc-print-art ${className}`} src={`/art/print/${kind}.svg`} width="360" height="300" alt="" aria-hidden="true" loading="lazy" decoding="async" />;
}
export function HardwareArt({ kind = 'laptop' }: { kind?: 'laptop' | 'desktop' | 'gx10' }) {
  return <svg viewBox="0 0 160 125" fill="none" aria-hidden="true" className="hc-hardware-art">
    {kind === 'laptop' ? <><path d="M25 20h110v76H25z" className="hc-machine-case"/><path d="M31 26h98v62H31z" className="hc-machine-screen"/><path d="M25 96 7 108v6h146v-6l-18-12M7 108h146M61 99h38l5 5H56z"/><path d="M72 51h16v16H72zM76 46v5m8-5v5m-8 16v5m8-5v5M67 55h5m-5 8h5m16-8h5m-5 8h5"/></> : kind === 'desktop' ? <><path d="M47 9h64v105H47z" className="hc-machine-case"/><path d="M54 18h50M54 25h50M58 10v103M72 40h25v38H72zM73 94h21"/><circle cx="82" cy="101" r="2"/></> : <><path d="m22 63 54-26 62 29-55 29z" className="hc-machine-screen"/><path d="M22 63v25l61 30 55-27V66M83 95v23M31 84l40 20M31 78l40 20M98 98l8-4m6-3 8-4"/><path d="m66 60 12-6 18 8-12 6z"/></>}
  </svg>;
}
