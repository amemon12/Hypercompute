// The presence hook calls crypto.randomUUID, which browsers expose only on https and localhost. A phone opening the
// dev server over plain http on the LAN would crash without this; getRandomValues works everywhere.
if (typeof crypto !== "undefined" && typeof crypto.randomUUID !== "function") {
  const uuid = (): string => {
    const b = crypto.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  };
  Object.defineProperty(crypto, "randomUUID", { value: uuid, configurable: true });
}

export {};
