import "./lib/polyfill";
import { ConvexProvider, ConvexReactClient } from "convex/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const url = import.meta.env.VITE_CONVEX_URL as string | undefined;
const root = createRoot(document.getElementById("root")!);

if (!url) {
  root.render(
    <main className="fatal">
      <h1>Pool Room is not configured</h1>
      <p>
        Set <code>VITE_CONVEX_URL</code> (run <code>npx convex dev</code> once, it writes <code>.env.local</code>).
      </p>
    </main>,
  );
} else {
  const convex = new ConvexReactClient(url);
  root.render(
    <StrictMode>
      <ConvexProvider client={convex}>
        <App />
      </ConvexProvider>
    </StrictMode>,
  );
}
