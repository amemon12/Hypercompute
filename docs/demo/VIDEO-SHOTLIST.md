# Hypercompute video shot list

Target: a local two-to-three-minute recording. Use the [runbook](JUDGE-RUNBOOK.md) for the live setup and
[NUMBERS](../pitch/NUMBERS.md) for every displayed measurement. The current edit is a captioned recorded rehearsal;
it must not be described as a continuous live take.

| Segment | Shot | Caption or narration | Evidence |
|---|---|---|---|
| Opening | Actual hypercompute.co header and introduction, cropped to omit decorative hardware labels. | “Run big models together.” | Website capture |
| Benchmarks | Laptops tab, then Every machine. Keep conditions and intervals legible. | “Measured comparisons. Network and model determine whether pooling helps.” | NUMBERS 1b and 3e |
| Join | Host peer list after mac-a joins. A separate Mac camera shot can show the physical double-click. | “The Mac join command ran remotely in this rehearsal. The physical double-click was not filmed.” | `hypercompute-join.command` exit status and actual host peer list |
| Loading | Real loading-state and ready-state captures, edited using the run timestamps. | “Time-compressed reconstruction from recorded states. Actual deployment to ready: 350.4 seconds. Shown: 12 seconds. Speed-up: 29.2x.” | Fresh functional-run provenance in NUMBERS; include seeding separately |
| Layers | Ready model and the planner's layer bar, with all participating computers named. | “Each segment is the layer assignment for this run.” | Actual ready-state capture and NUMBERS recording provenance |
| Chat | Real answer from the prepared deployment. Do not type a fake answer into the page. | “Recorded response from the pool.” | Actual app capture and sanitized SDK run evidence |
| Close | Measured laptop benefit, then the GX10 decode limit. | “Faster when pooling avoids disk reads. Not faster for every model.” | NUMBERS 1b, 2 and 3e |

The recorded rehearsal seeded for 180 seconds before deployment. The gateway then took 350.428424 seconds to become ready; the rounded caption above comes from NUMBERS section 7. The montage identifies its reconstruction from captured states.

## Capture procedure

1. Prepare the pool and finish both SDK smoke examples before capturing the large model.
2. Use a dedicated OBS scene collection and profile. Capture an individual app/browser surface or a local edit
   browser source. Do not use whole-desktop capture. Disable microphone and desktop audio unless narration is
   deliberately recorded.
3. Capture the actual website and its benchmark tabs. The decorative GPU artwork is illustrative and has hardware
   labels that are not measurements in NUMBERS, so crop it from the film.
4. Capture the host peer list after the real join. This rehearsal uses the actual macOS `.command` script over SSH;
   it does not supply footage of a person double-clicking the file. Keep that distinction on screen.
5. Capture the real loading state, ready state, layer bar and chat. Keep raw diagnostic output outside the recorded
   surface. Add measured loading metadata to NUMBERS before adding the time-compression caption.
6. Build the captioned edit from those captures. Use a persistent “Recorded rehearsal” label. Avoid transitions
   that make still captures look like a continuous live session.
7. Record the edit with OBS to a local file. Check the final file's duration, play representative frames from every
   segment, and confirm that the captioned loading times match the run record.
8. Keep the video outside the Git repository. Record the path, duration and SHA-256 in CROSS-CHECK after completion.
   Uploading or publishing requires approval.

## Secret and accuracy check

- The app removes the launch token from the address bar. Its invite input is masked. Neither measure makes it safe
  to record a terminal, clipboard history, browser developer tools or configuration files.
- Exclude the entire invite area when it could contain a ticket or code. If one appears in a raw take, crop or blur
  it before the final export and recheck the affected frames.
- Use generic peer aliases. Hide account names, home paths and unrelated notifications.
- Every displayed performance number, live snapshot value, layer count and load-timing label must have a source
  in NUMBERS. The recording-provenance section identifies functional checks separately from controlled benchmarks.
- Do not compare this rehearsal's token rate against older benchmark runs. Use the measured comparisons in the
  sheet and retain their model, host, network, prompt length and sample-size conditions.
- Do not claim pooled GX10 decode is always faster. The later interleaved 671B comparison is at parity; the earlier
  GX10 comparisons were slower pooled. Maverick's optional fast demo is GX10 alone.

For a later reshoot, replace the explicitly labeled remote-join segment with an actual Mac screen/camera take.
Leave the private invite outside the recorded region throughout the double-click and join.
