# Demo app theme: the hypercompute.co design system

Extracted on 2026-10-25 from the source of the live site so the demo app can look exactly like it.
Every value below is copied from a file; the file is named next to it.

- Site source: `frontend/website/` (Vite, React and TypeScript). These design notes were captured at commit `c2d6134`. The build output is `dist/`; the deployment configuration is `vercel.json`.
- Abbreviation used below: `WEB` = that folder.
- What is actually mounted on hypercompute.co (from `WEB/src/App.tsx` and the import graph):
  - `/`: `SiteHeader` (with `ScrollBrand` logo), hero, `SessionSection` (+ `OpenComputeDiagram`), `ExecutionDiagram` (+ `InferenceTimeline`), `SiteFooter` (+ `PortraitWordmark`).
  - `/benchmarks`: `BenchmarksContent` (+ `LargeModelResults`, `PoolingResults`).
  - Not mounted, but same design language and directly useful for the demo: `WEB/src/live/*` (`live.css` has the squared buttons, inputs, peer rows, invite block, chat and metrics), `MemoryPlan`, `CostEnergyEstimate`, `RuntimeProjection`, `ComputeShift`, `PrintArt`.
- The site is light only (`color-scheme:light` in `WEB/src/landing/landing.css`). The only dark-mode rule is inside the favicon SVG. The demo should stay light.

---

## 1. Files copied into the demo app

All copies are byte-identical (checked with `cmp`) except `fonts/fonts.css`, whose `url(/fonts/...)` paths were made relative.

| Demo path (`app/static/brand/`) | Copied from | What it is |
|---|---|---|
| `hypercompute-mark.svg` | `WEB/public/favicon.svg` | The chip mark (logo symbol). 72x72 viewBox, ink `#171714`, turns `#f4efe5` in dark mode. |
| `favicon.svg` | `WEB/public/favicon.svg` | Same file, under its site name, for `<link rel="icon">`. |
| `favicon-32.png` | `WEB/public/favicon-32.png` | 32x32 PNG fallback. |
| `apple-touch-icon.png` | `WEB/public/apple-touch-icon.png` | 180x180, ink chip on paper `#f4efe5`. |
| `paper-grain.svg` | `WEB/public/brand/paper-grain.svg` | 384x384 tiling paper noise. Required for the exact page background. |
| `paper-stains.svg` | `WEB/public/brand/paper-stains.svg` | 1600x3600 faint coffee-stain overlay, stretched over the page. |
| `hypercompute-tall-mask.svg` | `WEB/public/brand/hypercompute-tall-mask.svg` | Anton glyph outlines of "hypercompute", white fill, used as a CSS mask for the footer portrait wordmark. |
| `hypercompute-collage-mask.svg` | `WEB/public/brand/hypercompute-collage-mask.svg` | Geist Black outlines of "hypercompute", white fill. Shipped but not referenced by current pages. |
| `hypercompute-logo.webp` | `WEB/public/brand/hypercompute-logo.webp` | 2172x724 owner-supplied lockup (GPU card drawing + rounded "hypercompute" wordmark, black on white). Not used on the live site. Use only if a raster lockup is needed. |
| `fonts/anton-latin-400.woff2` | `WEB/public/fonts/` | Anton 400, Latin subset. |
| `fonts/geist-latin-400-500.woff2` | `WEB/public/fonts/` | Geist 400 and 500 (one official file serves both). |
| `fonts/geist-latin-900.ttf` | `WEB/public/fonts/` | Geist Black, outline source for wordmark masks only. |
| `fonts/ibm-plex-mono-latin-400.woff2` | `WEB/public/fonts/` | IBM Plex Mono 400, Latin subset. |
| `fonts/fonts.css` | `WEB/public/fonts/fonts.css` | `@font-face` rules, URLs now relative to the CSS file. |
| `fonts/*-OFL.txt`, `fonts/sources.txt` | `WEB/public/fonts/` | SIL OFL licenses and the exact Google Fonts URLs. Keep them next to the fonts. |

Not copied (large or people-specific), reference by path if ever needed:
- `WEB/public/brand/gpu-retro-paper.png` (2230x1264, 3.8 MB): the hero GPU print.
- `WEB/public/brand/gpu-clean-v2.webp`, `gpu-hardware-cutout.png`: other GPU art.
- Portrait JPGs for the footer wordmark (licenses in `WEB/public/brand/credits.txt`).
- `WEB/public/art/*.svg`, `WEB/public/art/print/*.svg`: print ornaments (unmounted).

---

## 2. Fonts

Loading on the site: `WEB/index.html` has `<link rel="stylesheet" href="/fonts/fonts.css" />`. All three web fonts are self-hosted Latin subsets with `font-display: swap`. Original Google Fonts CSS URL (from `sources.txt`):
`https://fonts.googleapis.com/css2?family=Anton&family=Geist:wght@400;500&family=IBM+Plex+Mono:wght@400&display=swap`

In the demo, load the vendored copy, no network needed:

```html
<link rel="stylesheet" href="/static/brand/fonts/fonts.css" />
```

| Role | Stack (exact) | Weights used | Where on the site |
|---|---|---|---|
| Body / UI text | `Geist, Arial, sans-serif` | 400, 500 | `landing.css` `.hc-page{font:400 16px/1.5 Geist,Arial,sans-serif}` |
| Headlines, logo, big numbers | `'Helvetica Neue', Arial, sans-serif` | 400, 500, 600, 700 | hero title, section h2, logo, benchmark numbers |
| Editorial ledes (serif) | `Georgia, 'Times New Roman', serif` | 400 | hero lede, section body copy, benchmark deck |
| Labels, data, units, layer ranges | `'IBM Plex Mono', monospace` | 400 | layer counts, table heads, tok/s units, captions |
| Footer wordmark only | `Anton` (via SVG mask, not live text) | 400 | `portrait-wordmark.css` |

Notes:
- Helvetica Neue and Georgia are system fonts and are not shipped. On Windows, Helvetica Neue is normally missing, so headlines and the logo render in Arial bold. That is how hypercompute.co looks on Windows; keep the same stack so the demo matches on each OS.
- Geist is declared only at 400 and 500. Do not ask Geist for 600 or 700; bold UI emphasis on the site uses Helvetica Neue or weight 500.
- `font-variant-numeric: tabular-nums` is used for numbers in rows (`benchmarks.css`, `live.css`). Use it for tok/s, GiB and layer counts.

---

## 3. Color

### Core tokens (defined on `#hc-home` in `WEB/src/landing/landing.css`)

| Token | Value | Use |
|---|---|---|
| `--hc-paper` | `#f4efe5` | Page background, `html`/`body` background, `<meta name="theme-color">` |
| `--hc-ink` | `#171713` | All text, strokes, outlines |
| `--hc-muted` | `#5d5e51` | Secondary text, link hover color |
| `--hc-green` / `--hc-accent` | `#4d7548` | Accent, focus outline, primary button fill |
| `--hc-soft` | `color-mix(in srgb, #4d7548 9%, #f4efe5)` | Pale green tint for selected states |
| `--hc-neutral` | `#dedccf` | Inactive layer blocks, empty fills |
| `--hc-paper-texture` | `url('/brand/paper-grain.svg')` | Paper grain tile |
| selection | bg `#c8d5bc`, text `#171713` | `WEB/src/styles.css` `::selection` |

### Extra tokens from the session UI (`WEB/src/live/live.css`, fallbacks used when the var is missing)

| Name | Value | Use |
|---|---|---|
| `--hc-green` fallback | `#374d38` | Darker green; button fill when `--hc-green` is unset |
| `--hc-copper` | `#9a5944` | Setup callout left border, reserve memory |
| `--hc-maroon` | `#733c3e` | Errors (left border + text), "does not fit" verdict |
| button text on green | `#f6f3e9` | Primary button label |
| `--hc-muted` fallback | `#68685d` | Fine print |
| offline dot | `#8d8277` | Status dot when not connected |

### Device and layer colors (`WEB/src/landing/execution-diagram.css`, `InferenceTimeline.tsx`)

This is the "which machine holds which layers" palette. Machine order is fixed: first, second, third.

| Machine | Block fill | Bar stroke when active | Token on site |
|---|---|---|---|
| 1 | `#98ad92` (sage) | `#5d7657` | `--hd-green` |
| 2 | `#d69876` (copper) | `#a46744` | `--hd-copper` |
| 3 | `#ccc9bd` (stone) | `#8d897a` | `--hd-neutral` |

The site never shows more than three machines. For a 4th to 6th peer, extend with colors that already appear on mounted pages: `#9bb7c6` (blue board, `OpenComputeDiagram.tsx`), `#77916e` (pooled bar green, `large-model-results.css`), `#bb8057` (chart copper, `BenchmarksContent.tsx`). Ink text stays readable on all of them (ink on `#98ad92` 7.47:1, on `#d69876` 7.37:1, on `#ccc9bd` 10.83:1).

### Rules, lines and chart colors

| Value | Where | Use |
|---|---|---|
| `#c8c2b5` | `--hd-rule`, execution-diagram.css | Row dividers, layer brackets |
| `#b9b5a8` | live.css | Input borders, section rules, peer list top rules |
| `#d0ccbf` | live.css | Light row dividers (peer rows, chat turns) |
| `#a9a697` | live.css | Quiet button border, metric top rules |
| `#a6a092` | large-model-results.css | Select underline |
| `#bdb8aa` | InferenceTimeline.tsx | Timeline grid lines (opacity .38 / .55) |
| `#68675e` | InferenceTimeline.tsx | Timeline arrows |
| `#5b5a52` | InferenceTimeline.tsx | Timeline `t0..t3` labels |
| `#e5e1d6` | execution-diagram.css | Screen fill inside device drawings |
| `#77916e` / `#c5c1b5` | large-model-results.css | Pooled vs single-machine bars |
| `#c78251`, `#7c9b68` | SessionSection.tsx | Print-corner ornaments (copper block, green curve) |
| `#949084`, `#79776c`, `#5278ab` | landing.css | GPU pointer lines, dots, active dot (blue) |

### Contrast on paper `#f4efe5` (WCAG, computed)

ink 15.68, muted `#5d5e51` 5.75, green `#4d7548` 4.63, maroon 7.50, copper `#9a5944` 4.72, `#68685d` 4.92. Button text `#f6f3e9` on `#4d7548` 4.78, on `#374d38` 8.29. `#8d8277` is 3.28 (dots only, never text). `#b9b5a8` is 1.79 (rules only).

### Ready-to-paste tokens for the demo

```css
:root{
  --hc-paper:#f4efe5; --hc-ink:#171713; --hc-muted:#5d5e51;
  --hc-green:#4d7548; --hc-accent:#4d7548; --hc-green-deep:#374d38;
  --hc-soft:color-mix(in srgb,#4d7548 9%,#f4efe5); --hc-neutral:#dedccf;
  --hc-copper:#9a5944; --hc-maroon:#733c3e; --hc-on-green:#f6f3e9;
  --hd-green:#98ad92; --hd-copper:#d69876; --hd-neutral:#ccc9bd; --hd-rule:#c8c2b5;
  --hc-rule:#b9b5a8; --hc-rule-light:#d0ccbf; --hc-rule-strong:#a9a697;
  --hc-sans:Geist,Arial,sans-serif;
  --hc-display:'Helvetica Neue',Arial,sans-serif;
  --hc-serif:Georgia,'Times New Roman',serif;
  --hc-mono:'IBM Plex Mono',monospace;
  color-scheme:light;
}
```

---

## 4. Page background (paper)

From `landing.css` and `styles.css`:

```css
html{background:#f4efe5}
body{margin:0;color:#171713;background:#f4efe5}
.hc-page{position:relative;background:#f4efe5 url('brand/paper-grain.svg') repeat;background-size:384px 384px;color:#171713;font:400 16px/1.5 Geist,Arial,sans-serif}
.hc-page::before{content:'';position:absolute;inset:0;z-index:0;pointer-events:none;background:url('brand/paper-stains.svg') center top/100% 100% no-repeat}
/* <=760px */ .hc-page::before{background-size:900px 2600px;background-position:right top}
```

Paths above are relative to `app/static/`; the site uses `/brand/...`. Put content above the `::before` layer with `position:relative`.

---

## 5. Type scale

| Element | Font | Weight | Size | Line height | Tracking | Source |
|---|---|---|---|---|---|---|
| Logo wordmark | Helvetica Neue | 700 | 36px (30px at <=760px) | 1 | -.06em | landing.css `.hc-logo` |
| Hero title | Helvetica Neue | 700 | clamp(32px, 3.35cqw, 50px) | 1.12 | -.035em | `.hc-title` |
| Hero lede | Georgia | 400 | clamp(22px, 2.05cqw, 30px), max-width 760px | 1.4 | -.025em | `.hc-lede` |
| Section headline | Helvetica Neue | 700 | clamp(38px, 3.8cqw, 60px) | 1.06 | -.045em | session-section.css h2 |
| Section body | Georgia | 400 | clamp(22px, 2.05cqw, 30px) | 1.42 | -.018em | session-section.css p |
| Row heading | Helvetica Neue | 400 | clamp(24px, 2.2cqw, 31px) | 1.16 | -.025em | execution-diagram.css `.hd-copy h2` |
| Row body | Georgia | 400 | clamp(16px, 1.4cqw, 20px) | 1.4 | -.01em | `.hd-copy p` |
| Page title (benchmarks) | Helvetica Neue | 700 | clamp(44px, 5.6vw, 76px) | 1.04 | -.055em | benchmarks.css h1 |
| Deck under title | Georgia | 400 | clamp(26px, 2.65vw, 36px) | 1.25 | -.022em | `.hc-benchmarks-deck` |
| Section h2 (benchmarks) | Helvetica Neue | 600 | 26px | 1.2 | -.025em | benchmarks.css h2 |
| Big metric number | inherits Geist | 500 | clamp(30px, 3.2vw, 42px) | 1.15 | -.045em | `.hc-benchmark-median` |
| Table metric | inherits Geist | 500 | clamp(24px, 2.5vw, 33px) | default | -.035em | `.hc-large-table td strong` |
| Live metric number | inherits | 400 | clamp(26px, 3vw, 42px) | 1.15 | -.045em | live.css `.hc-live-metrics strong` |
| Model / device name | Geist | 500 | 18 to 19px | 1.3 to 1.35 | -.02em | `.hc-benchmark-model`, `.hc-large-model` |
| Device name in diagram | Geist | 400 | 18px | 1.3 | 0 | `.hd-device-name` |
| Body | Geist | 400 | 16px | 1.5 (1.6 in long text) | 0 | `.hc-page` |
| Nav link | Geist | 400 | 15px (13px mobile) | 1.3 | 0 | `.hc-section-nav a` |
| Small print / captions | Geist | 400 | 12 to 14px | 1.5 to 1.65 | 0 | captions, notes |
| Mono label | IBM Plex Mono | 400 | 11 to 13px | 1.4 to 1.5 | 0 (.025em on GPU labels) | table heads, units, ranges |
| Mono caps label | IBM Plex Mono | 400 | 13px, `text-transform:uppercase` | 1.4 | .05em | `.hd-layer-total` ("48 LAYERS") |
| Live small caps label | inherits | 400 | 12px, uppercase | default | .05em | live.css `.hc-live-label` |

Headings and paragraphs have `margin:0`; spacing comes from explicit `margin-top`.

---

## 6. Layout and spacing

- Content max width 1440px (`.hc-content`); benchmark text column `width:min(1200px, calc(100% - 96px))`, padding `56px 0 104px`.
- Sticky header: height 96px (116px at <=760px), `background:inherit`, z-index 20. Inner grid `grid-template-columns:minmax(0,1fr) auto minmax(0,1fr)`, gap 24px, padding `0 40px`. Left nav, centered logo, right nav. Nav gap 32px. At <=760px the logo takes row 1 and two nav groups share row 2.
- Section padding: 48px sides (64px when container >=1200px, 32px <=900px, 24px <=760px, 22px <=480px). Hero padding `clamp(28px,3cqw,40px) 48px 32px`, gap 28px.
- Two-column rows (`.hd-row`): `grid-template-columns:minmax(200px,.8fr) minmax(0,2fr)`, gap 48px, padding 36px 0, `border-top:1px solid #c8c2b5` between rows. One column at <=760px.
- Footer: padding `48px 24px 32px`, gap 48px, info width 1280px max, nav font `clamp(18px,1.7vw,24px)`, meta 14px.
- Rhythm actually used: 4, 8, 10, 12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 96. Default steps: 8 / 12 / 16 / 24 / 32 / 48 / 64.
- Breakpoints: media 760px and 480px; container queries at 1200px (min), 900px, 560px, 360px; benchmarks also 900px and 600px.
- Touch targets: links and buttons keep `min-height:44px` (46px for live buttons).

---

## 7. Shape, borders, shadows

- Radius: zero. Every control sets `border-radius:0` (live buttons, inputs, selects, memory buttons). Rounding only exists inside drawings: chip mark corners `rx 4` in a 72 box, compact logo die 3px, timeline bars `rx 2`, device cases `rx 3`, and 4 to 7px round dots.
- Shadows: none. There is no `box-shadow`, `text-shadow` or `drop-shadow` anywhere in the site CSS.
- No cards. Groups are separated by 1px rules (`#c8c2b5`, `#b9b5a8`, `#d0ccbf`) and callouts use a left border: invite `2px solid green` + `padding-left:24px`; setup `3px solid copper`; error `3px solid maroon`.
- Square status dot: `7px x 7px`, no radius, `#8d8277` offline, green online.

---

## 8. Motion

- Logo morph (`scroll-brand.css`): `max-width` and `transform` `.62s cubic-bezier(.65,0,.2,1)`, opacity `.25s ease .12s`; die `.38s ease`; pins `opacity .15s ease, transform .28s ease` with delays .14s / .16s / .32s.
- Layer blocks: `background .18s ease`; allocation outline `outline-color .2s ease`; memory buttons `background .2s`.
- Scroll reveal (`compute-shift.css`): `translateY(10px)` to `0`, scroll-timeline driven, only under `prefers-reduced-motion:no-preference`.
- Smooth anchor scrolling: `html{scroll-behavior:smooth}`.
- Reduced motion (`styles.css`): all animation and transition durations forced to `.01ms`, smooth scroll off. The demo must honor the same rule.
- Motion is small and slow. No bounces, no spinners on the site. For "working" states prefer a calm opacity pulse or a filling bar.

---

## 9. Logo

### 9a. Header wordmark (what visitors see on hypercompute.co)

Live text, not an image: lowercase "hypercompute" in Helvetica Neue 700, 36px, tracking -.06em, ink, centered in a 248px slot. Source `WEB/src/landing/ScrollBrand.tsx` + `scroll-brand.css` + `landing.css .hc-logo`. Markup:

```html
<a class="hc-logo" href="#" aria-label="Hypercompute home" data-compact="false">
  <span class="hc-brand-word" aria-hidden="true">
    <span class="hc-brand-prefix"><span>hyperc</span></span>
    <span class="hc-brand-core">
      <span class="hc-brand-o">o</span>
      <span class="hc-brand-die"></span>
      <svg class="hc-brand-pins" viewBox="0 0 40 44" focusable="false">
        <path d="M14 10V5 M20 10V5 M26 10V5 M14 34V39 M20 34V39 M26 34V39 M8 16H3 M8 22H3 M8 28H3 M32 16H37 M32 22H37 M32 28H37" />
      </svg>
    </span>
    <span class="hc-brand-suffix"><span>mpute</span></span>
  </span>
</a>
```

CSS (site rules with the `#hc-home` scope removed):

```css
.hc-logo{display:flex;align-items:center;justify-content:center;flex:none;width:248px;min-height:48px;color:#171713;font:700 36px/1 'Helvetica Neue',Arial,sans-serif;letter-spacing:-.06em;white-space:nowrap;text-decoration:none}
.hc-brand-word{display:inline-flex;align-items:center;height:44px;white-space:nowrap}
.hc-brand-prefix,.hc-brand-suffix{display:block;overflow:hidden;transition:max-width .62s cubic-bezier(.65,0,.2,1)}
.hc-brand-prefix{max-width:3.3em}
.hc-brand-suffix{max-width:3.5em}
.hc-brand-prefix>span,.hc-brand-suffix>span{display:block;transition:transform .62s cubic-bezier(.65,0,.2,1),opacity .25s ease .12s}
.hc-brand-core{position:relative;display:inline-flex;align-items:center;justify-content:center;flex:none;width:.56em;height:44px;transition:width .62s cubic-bezier(.65,0,.2,1)}
.hc-brand-o{transition:opacity .16s ease .3s}
.hc-brand-die{position:absolute;left:50%;top:50%;width:.53em;height:.62em;border:.1em solid currentColor;border-radius:50%;opacity:0;transform:translate(-50%,-50%);transition:width .38s ease,height .38s ease,border-radius .38s ease,opacity .15s ease}
.hc-brand-pins{position:absolute;left:50%;top:50%;width:40px;height:44px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:square;opacity:0;transform:translate(-50%,-50%) scale(.6);transition:opacity .15s ease,transform .28s ease}
.hc-logo[data-compact=true] .hc-brand-prefix,.hc-logo[data-compact=true] .hc-brand-suffix{max-width:0}
.hc-logo[data-compact=true] .hc-brand-prefix>span{transform:translateX(100%);opacity:0;transition-delay:0s}
.hc-logo[data-compact=true] .hc-brand-suffix>span{transform:translateX(-100%);opacity:0;transition-delay:0s}
.hc-logo[data-compact=true] .hc-brand-core{width:40px}
.hc-logo[data-compact=true] .hc-brand-o{opacity:0;transition-delay:.14s}
.hc-logo[data-compact=true] .hc-brand-die{width:24px;height:24px;border-radius:3px;opacity:1;transition-delay:.16s}
.hc-logo[data-compact=true] .hc-brand-pins{opacity:1;transform:translate(-50%,-50%) scale(1);transition-delay:.32s}
@media(prefers-reduced-motion:reduce){.hc-brand-word *{transition:none!important}}
```

Behavior on the site: `data-compact` becomes `true` once `scrollY > 64` (back to false below 16). The "o" turns into a chip with pins and the letters slide away. In the demo the same toggle is a nice "pool is live" moment, for example compact while a model is loading.

### 9b. Chip mark (logo symbol and favicon)

File: `app/static/brand/hypercompute-mark.svg` (copy of `WEB/public/favicon.svg`). Full source:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72"><style>path{fill:#171714}@media (prefers-color-scheme:dark){path{fill:#f4efe5}}</style><path fill-rule="evenodd" d="M16 12h40a4 4 0 0 1 4 4v40a4 4 0 0 1-4 4H16a4 4 0 0 1-4-4V16a4 4 0 0 1 4-4zM19 19v34h34V19z"/><path d="M21.5 0h4v13h-4zM34 0h4v13h-4zM46.5 0h4v13h-4zM21.5 59h4v13h-4zM34 59h4v13h-4zM46.5 59h4v13h-4zM0 21.5h13v4H0zM0 34h13v4H0zM0 46.5h13v4H0zM59 21.5h13v4H59zM59 34h13v4H59zM59 46.5h13v4H59z"/></svg>
```

### 9c. Footer wordmark

`PortraitWordmark` fills a full-width block with six grayscale portraits and clips it with `hypercompute-tall-mask.svg` (Anton outlines, `aspect-ratio:12106.336/2311.5`, background `#181818`, `mask-size:100% 100%`, `mix-blend-mode:multiply`). For the demo, use the mask with a flat ink fill instead of portraits:

```css
.hc-footer-wordmark{width:100%;aspect-ratio:12106.336/2311.5;background:#171713;-webkit-mask:url('brand/hypercompute-tall-mask.svg') center/100% 100% no-repeat;mask:url('brand/hypercompute-tall-mask.svg') center/100% 100% no-repeat}
```

### 9d. Raster lockup

`hypercompute-logo.webp` is owner-supplied and not shown on the live site. It uses a different rounded typeface and a white background, so it will not match the paper page. Prefer 9a + 9b.

---

## 10. Favicon and head tags (from `WEB/index.html`)

```html
<meta name="theme-color" content="#f4efe5" />
<link rel="stylesheet" href="/static/brand/fonts/fonts.css" />
<link rel="icon" type="image/png" sizes="32x32" href="/static/brand/favicon-32.png" />
<link rel="icon" type="image/svg+xml" href="/static/brand/favicon.svg" />
<link rel="apple-touch-icon" href="/static/brand/apple-touch-icon.png" />
<title>Hypercompute · Demo</title>
```

The site title uses an em dash as separator. The demo must not; use the middle dot, which the site itself uses in labels ("First token · median", "Windows · PowerShell").

---

## 11. Components

### Links and nav
- Text links: ink, no underline; hover color `#5d5e51` (landing) or underline `1px`, `text-underline-offset:5px to 6px`.
- Nav links: 15px, `min-height:44px`, hover and `aria-current=page` underline 1px, offset 6px.
- Focus: `outline:2px solid #4d7548; outline-offset:5px` (landing) or `outline-offset:4px` (live).

### Buttons (from `live.css`, the only button system in the codebase)

```css
.hc-button{display:inline-flex;gap:26px;justify-content:space-between;align-items:center;min-height:46px;padding:12px 18px;border:1px solid var(--hc-green);border-radius:0;background:var(--hc-green);color:#f6f3e9;cursor:pointer;text-align:left;line-height:1.25;font:inherit}
.hc-button:hover:not(:disabled){background:var(--hc-ink);border-color:var(--hc-ink)}
.hc-button-quiet{background:transparent;color:var(--hc-ink);border-color:#a9a697}
.hc-button-quiet:hover:not(:disabled){color:#f6f3e9}
.hc-button:disabled{opacity:.48;cursor:not-allowed}
.hc-text-button{border:0;background:none;padding:10px 0;color:inherit;text-decoration:underline;text-underline-offset:5px;min-height:44px;cursor:pointer}
```

Primary is a flat green rectangle that turns ink on hover. The `gap:26px` and `justify-content:space-between` leave room for a trailing arrow or chip glyph. Secondary is the quiet outline. Tertiary is an underlined text button.

Segmented toggle (`.hc-live-mode`): row of text buttons, `gap:28px`, `border-bottom:1px solid #b9b5a8`; selected gets `border-bottom:3px solid green` and weight 600.

Pressed chip buttons (`memory-plan.css`, unmounted): 46px tall, `padding:11px 22px`, square. Their blue fills are off-palette; do not use.

### Inputs
- Underline style (mounted, benchmarks): `border:0;border-bottom:1px solid #a6a092;border-radius:0;background:transparent;padding:10px 28px 10px 0;font:400 15px/1.5 Geist`. Label above in Plex Mono 12px, gap 8px.
- Boxed style (live.css): `border:1px solid #b9b5a8;border-radius:0;background:transparent;padding:12px;min-height:46px`; 16px font on mobile to stop iOS zoom. Checkboxes use `accent-color` green, 18px.

### Peer / machine rows (live.css `.hc-live-node`)
`display:grid;grid-template-columns:20px minmax(0,1fr) auto;gap:16px;padding:16px 0;border-bottom:1px solid #d0ccbf`. Name `strong` weight 400, detail `small` 13px muted, value column right aligned with tabular numbers. Empty state copy on the site: "Computers appear here when they connect."

### Device drawing (execution-diagram.css + `ExecutionDiagram.tsx`)
Device button: transparent, centered column, drawing box 160px wide by 130px tall, name 18px, range in Plex Mono 13px ("Layers 00–15"). Selected: name underlined (offset 5px). Devices are joined by a 1px `#c8c2b5` line. SVG source, `viewBox="0 0 120 94"`, styles `svg{stroke:#171713;stroke-width:1.25;stroke-linejoin:round;stroke-linecap:round} .hd-case{fill:#f4efe5} .hd-display{fill:#e5e1d6;stroke:none}`:

```html
<!-- Laptop -->
<svg viewBox="0 0 120 94" aria-hidden="true" focusable="false">
  <rect class="hd-case" x="15" y="21" width="90" height="62" rx="3"/>
  <rect class="hd-display" x="19" y="25" width="82" height="54" rx="1"/>
  <path class="hd-case" d="M5 84h110v3c0 4-6 6-11 6H16C11 93 5 91 5 87z"/>
  <path d="M47 84h26l-3 3H50z" fill="none"/>
</svg>
<!-- Desktop tower -->
<svg viewBox="0 0 120 94" aria-hidden="true" focusable="false">
  <rect class="hd-case" x="34" y="2" width="52" height="82" rx="3"/>
  <path d="M45 9v68M55 13h22m-22 6h22m-22 6h22" fill="none"/>
  <rect class="hd-display" x="55" y="35" width="22" height="27"/>
  <path d="M72 75h5" fill="none"/>
</svg>
```

### Layer split (the "which machine holds which layers" figure)
From `execution-diagram.css`:
- Header: `"48 LAYERS"` in Plex Mono 13px uppercase .05em, centered between two 8px bracket lines (`border-top:1px solid #c8c2b5` with end ticks).
- Groups: `display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px`. Make the columns proportional to each machine's layer count in the demo.
- Blocks: one `<i>` per layer, `display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:2px;height:34px` (30px and gap 1px when narrow), each block filled with the machine color.
- Active group: `outline:1px solid #171713;outline-offset:3px` (transition .2s).
- Under each group: bracket `height:8px;border:solid #c8c2b5;border-width:0 1px 1px` with a 7px center tick, then machine name 16px/1.4 and count in Plex Mono 12px ("16 layers").

### Metrics (tok/s and friends)
- Benchmarks style: number in Geist 500 clamp(30px,3.2vw,42px), tracking -.045em, unit underneath or beside in Plex Mono 11px ("tokens/s").
- Live style: three columns, `gap:24px`, each with `border-top:1px solid #a9a697;padding-top:16px`, label 12px, number clamp(26px,3vw,42px) weight 400 tracking -.045em.
- Bars: 22px tall flat fills, value in Plex Mono 12px to the right.

### Chat (live.css)
Log `max-height:540px;overflow:auto;overscroll-behavior:contain`. Each turn `padding:20px 0;border-top:1px solid #d0ccbf`; question weight 600; answer `white-space:pre-wrap;overflow-wrap:anywhere`. Section starts with `padding-top:36px;border-top:1px solid #b9b5a8`.

### Invite block (live.css `.hc-invite`)
`border-left:2px solid green;padding-left:24px`; a `dl` of label 13px over value. For the demo: never render the ticket or code text itself, show a masked placeholder with a Copy button.

### Status line
`display:flex;gap:10px;font-size:13px`, square 7px dot first. Errors: `padding:12px 0 12px 18px;border-left:3px solid #733c3e;color:#733c3e`.

### Accessibility bits
Skip link `position:fixed;top:8px;left:16px;padding:12px 16px;background:#171713;color:#f4efe5`, hidden with `translateY(-160%)` until focus. `button:disabled{opacity:.55;cursor:not-allowed}` globally. Tap highlight removed (`-webkit-tap-highlight-color:transparent`).

---

## 12. Copy voice

How the site writes (examples are verbatim from mounted pages and the session UI):

- Headlines are short full sentences with a period. "We’re building multiplayer computing." "Runs where your model lives." "What ran on our laptops."
- Ledes explain the mechanism in one plain sentence. "Start a session on one computer and connect the others. Hypercompute splits the model’s layers across their GPUs over your local network."
- Second person and concrete nouns: computers, GPUs, layers, memory, model. "Each computer holds part of the model."
- Buttons are verbs in sentence case: "Start the host", "Create a session", "Contribute this computer", "Connect this computer", "Load the model", "Pause sharing", "Resume sharing", "Leave session".
- Honest and measured. Numbers come with conditions. "Recorded timings only." "The split depends on available memory." "These figures describe this chat session; they do not establish a speedup over a single computer."
- Units and data in mono: `tokens/s` with one decimal, `GiB`, `ms`, `Layers 00–15`. Middle dot joins label pairs: "Range · samples".
- Real hardware names: "ASUS ROG Zephyrus G14", "M4 MacBook Air", "RTX 5070 Ti Laptop GPU".
- No exclamation marks, no emoji, no hype words. Curly apostrophes (’) in prose.
- Dashes: numeric ranges use an en dash (`00–15`). The site has a few em dashes (page title, one benchmark note). The demo uses none. Use a period, comma, colon or middle dot instead.

Demo copy in this voice, for reference: "Host a pool" / "Join a pool" / "Copy invite" / "Computers appear here when they join." / "Pick a model" / "Plan and load" / "Ask the pool" / "Stop sharing". Status lines like "3 computers · 41.5 GiB shared" and "12.4 tokens/s".

---

## 13. Quick checklist for the demo UI

1. Paper `#f4efe5` with grain + stains, ink `#171713`, green `#4d7548` as the single accent.
2. Geist body, Helvetica Neue headlines, Georgia ledes, IBM Plex Mono for numbers and labels.
3. Header: three-column grid, centered lowercase wordmark, 96px tall, sticky.
4. Square corners, 1px rules, left-border callouts, no shadows, no cards.
5. Layer split uses sage, copper, stone blocks, 2px gaps, 34px tall.
6. Light only, reduced motion honored, 44px targets, 2px green focus ring.
7. No em dashes anywhere.
