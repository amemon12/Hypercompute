# Hypercompute website

The public-facing React/Vite website lives in `frontend/website/`. The existing inference dashboard remains in `frontend/src/`, with its own package and build.

## Run locally

From the repository root:

```sh
cd frontend/website
npm ci
npm run dev
```

Open http://127.0.0.1:3000. The homepage explains shared computing. `/benchmarks` shows three comparisons: generation speed, time to first response, and total output across simultaneous chats. Expand Run details for the hardware, workload, and source information. Both pages work without a local inference host.

The selected Kimi and Qwen values confirmed by the project owner on October 25, 2026 are documented as [team-reported website benchmark values](../../docs/RESULTS.md#website-benchmark-values). Archived measurements remain available in the same results document.

## Build and check

```sh
npm run build
npm test
```

The build type-checks the source and writes the production website to `dist/`. Build output and `node_modules/` are ignored by Git. Calculator tests require Node.js with built-in TypeScript stripping (the cross-check used the installed Node runtime).

## Source layout

- `src/landing/`: homepage, shared navigation/footer, diagrams, and benchmark components.
- `src/landing/benchmark-story.ts`: data for the three public comparisons, including the documented team-reported values.
- `src/landing/BenchmarksContent.tsx`: comparison charts, explanatory text, run details, and source links.
- `src/landing/benchmark-charts.ts`, `benchmark-sheet.ts`, and `benchmark-evidence.ts`: retained chart data, historical tables, and source links. The public story uses the selected team-reported values without rewriting those archived assignments.
- `src/live/`: retained local-session UI components and memory calculations; these are not mounted by the current public pages. They reuse the dashboard API client from `frontend/src/api/`.
- `public/`: artwork, local fonts, and attribution/license files.

Vite keeps fixed local proxies for `/api` and `/v1` on port 8000 and `/local-api` on port 8001 for session components. The native connector requires the corresponding backend implementation; uploading this frontend does not install or start it.

The website preserves the paper texture, green/copper accents, existing fonts, and portrait wordmark. Sections use natural page scrolling rather than screen-sized snap panels.
