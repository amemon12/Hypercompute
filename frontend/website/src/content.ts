export interface Article {
  slug: string;
  category: string;
  date: string;
  title: string;
  summary: string;
  art: 'network' | 'layers' | 'orbit' | 'memory';
  sections: { heading: string; body: string }[];
  source: string;
}

export const articles: Article[] = [
  {
    slug: 'introducing-hypercompute',
    category: 'Announcements',
    date: 'Project notes',
    title: 'Introducing Hypercompute',
    summary: 'Run a model across computers you trust.',
    art: 'network',
    sections: [
      {
        heading: 'Connect your computers',
        body: 'Hypercompute lets friends run one model across several computers. One person starts a session and invites others to contribute GPU memory. The host manages the model, while a small agent connects each contributing computer.',
      },
      {
        heading: 'See where the layers will run',
        body: 'Before a model starts, Hypercompute checks available memory and shows where its layers will run. After loading, it reads the inference engine’s own logs to verify that the selected computers actually received work. The dashboard brings the session, plan, chat, and run records into one place.',
      },
      {
        heading: 'Manage the model from the host',
        body: 'The model file stays on the host. Each contributor receives the layers it will run through an encrypted connection. Contributors can pause or stop sharing from their own machines. Hypercompute is designed for people who trust one another on a private network.',
      },
      {
        heading: 'Current implementation',
        body: 'We have run the SDK and one-click scripts on a GX10, a Windows laptop and two MacBooks. The planner assigns layers, the app shows their placement, and chat runs through the pool. Pooling helps some models and slows others; the benchmark page includes both.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/README.md',
  },
  {
    slug: 'engine-tiers',
    category: 'Announcements',
    date: 'Project notes',
    title: 'How a model fits the pool',
    summary: 'The planner decides where weights fit and checks the result before loading.',
    art: 'layers',
    sections: [
      {
        heading: 'How the planner chooses memory',
        body: 'Hypercompute’s default strategy first looks for a model placement that fits on GPUs. If more memory is needed, the planner considers peer GPUs and the host’s RAM. Slower disk-backed options are capacity fallbacks. Each plan makes the choice visible, and one model uses one engine at a time.',
      },
      {
        heading: 'Windows and macOS in one pool',
        body: 'The standard runtime divides model layers across local and remote devices. A Windows NVIDIA GPU can work with Apple Silicon on macOS. The host checks each worker’s compatibility before including it in a plan, then verifies that it received layers after loading.',
      },
      {
        heading: 'When memory runs out',
        body: 'The planner can use host RAM or keep some mixture-of-experts weights backed by a file on the host. Disk reads can make generation slow. The demo refuses full disk-paging plans on the GX10 because those loads have exhausted its memory in testing.',
      },
      {
        heading: 'Check the measured result',
        body: 'A plan estimates capacity and speed. The benchmark records show what happened on the hardware: load time, first token, decode rate and layer placement. A model that already fits one machine often runs faster there.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/docs/ENGINE_TIERS.md',
  },
  {
    slug: 'first-connected-session',
    category: 'Engineering',
    date: 'Project notes',
    title: 'Our first Windows-and-Mac session',
    summary: 'Layer placement and decode speeds from an RTX 5070 Ti laptop and an M4 MacBook Air.',
    art: 'orbit',
    sections: [
      {
        heading: 'The test setup',
        body: 'Our first physical pair session joined a Windows laptop with an RTX 5070 Ti and an M4 MacBook Air over campus Wi-Fi. The Mac completed a real RPC handshake and received model layers. Three deployments provided the first evidence beyond loopback workers and a Linux virtual machine on the host.',
      },
      {
        heading: 'Layer placement and decode speed',
        body: 'The planned splits matched the placements reported by the runtime. A Qwen2.5 14B Q6_K run placed two transformer layers and the output head on the Mac, generating between 11.7 and 12.3 tokens per second across two chats. Mistral Small 3.2 24B Q4_K_M placed eight layers and the head there, with a median of 9.1 tokens per second across twelve chats.',
      },
      {
        heading: 'Output-head placement',
        body: 'These runs used an earlier device order that put the output head on the Mac. That sent a large result tensor back across Wi-Fi for every generated token. The current planner keeps the head on the host’s GPU. Its effect on this physical pair still needs measurement, and the host’s power state was not recorded during the first session.',
      },
      {
        heading: 'Further tests',
        body: 'Later sessions measured larger models on laptop pools and a GX10 host. The results separate home-network runs from campus Wi-Fi runs. This first session establishes real participation; it does not establish a general speedup or prove that each model could not run on either computer alone.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/docs/IMPLEMENTATION_STATUS.md',
  },
  {
    slug: 'how-layer-splitting-works',
    category: 'Engineering',
    date: 'Technical notes',
    title: 'How layer splitting works',
    summary: 'How the host assigns layers and passes token computation between GPUs.',
    art: 'layers',
    sections: [
      {
        heading: 'Layer assignment',
        body: 'In Hypercompute’s default workflow, one server on the host owns the model. The model’s transformer layers are divided into contiguous groups assigned to selected GPUs. Each contributor runs an RPC worker that receives its assigned weights and computes those layers as tokens are generated.',
      },
      {
        heading: 'Capacity and network costs',
        body: 'For a single conversation, the layer groups form a sequence of work. Assigning some layers to another computer adds remote computation and network transfers. Hypercompute therefore fills the host GPU first when possible and considers other memory for the overflow. A model that already fits locally will not automatically benefit from splitting.',
      },
      {
        heading: 'Keep the output head on the host',
        body: 'The runtime assigns the output head to the last device in its layer-split order. Hypercompute places the host GPU last so the large vocabulary output can stay local. Keeping that output on a remote device would send substantially more data across the network for each token. The benefit on any particular setup still depends on measurement.',
      },
      {
        heading: 'Verify layer placement',
        body: 'A connected computer may still receive no model layers. Hypercompute checks worker compatibility before planning and inspects the engine’s loading logs afterward. The dashboard can show the layer counts and device buffers reported by the engine, so participation is tied to observed placement.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/docs/research/llama_placement.md',
  },
  {
    slug: 'local-control',
    category: 'Principles',
    date: 'Technical notes',
    title: 'Sharing limits and local controls',
    summary: 'How invitations, memory limits, and stop controls work on each computer.',
    art: 'memory',
    sections: [
      {
        heading: 'Invitations and memory limits',
        body: 'The host creates single-use invitations that expire after fifteen minutes. A joining agent reports its hardware and the memory it offers, then asks the contributor to confirm. The host verifies the worker before it can take part in a run. A memory budget guides planning; it is a soft budget, not a hard operating-system limit.',
      },
      {
        heading: 'Pause or stop sharing',
        body: 'Every contributor can pause, leave, stop the agent, or use the local Stop button. If a participating worker disappears, Hypercompute invalidates the deployment and records the reason. If the host disappears for five minutes, the contributor’s worker stops automatically. The host cannot use the agent to run arbitrary commands.',
      },
      {
        heading: 'Where model data is stored',
        body: 'Inference runs across the participating computers, and the dashboard runs on the host. The original GGUF file is held by the host; contributors receive the layers needed for their work. Optional layer caching can keep received data on a contributor’s disk, and other engines have their own model-download behavior.',
      },
      {
        heading: 'Use a network you trust',
        body: 'The one-click scripts use encrypted Iroh connections and keep workers on loopback. Raw worker RPC has no authentication of its own, so direct LAN mode needs a trusted network and firewall rules. Hypercompute does not change those rules for you.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/README.md#security-model',
  },
  {
    slug: 'measured-tuning',
    category: 'Engineering',
    date: 'Technical notes',
    title: 'Tuning a model for your hardware',
    summary: 'Compare placements and save a measured configuration for later runs.',
    art: 'network',
    sections: [
      {
        heading: 'Initial placement estimates',
        body: 'The planner uses model size, device memory, and a decode-cost model to choose a feasible placement. When a model exceeds the local GPU budget, a friend’s GPU and the host’s RAM can both be candidates. Network latency and device speed determine which is preferable, so a generic estimate cannot settle every case.',
      },
      {
        heading: 'Compare configurations',
        body: 'The hypercompute tune command loads candidate setups one after another. It measures time to first token, decode speed, and prompt processing while comparing choices such as overflow placement, flash attention, and micro-batch size. The measurements run without competing chat requests on that deployment.',
      },
      {
        heading: 'Reuse a measured profile',
        body: 'The tuner saves the best-performing setup as a measured profile. Later plans for the same model, context, and machines can reuse that profile instead of relying only on the initial cost estimate. The dashboard distinguishes a measured profile from an estimate, keeping the basis for the decision visible.',
      },
      {
        heading: 'Recorded tuning results',
        body: 'An early laptop test compared four placements and successfully reused the fastest measured profile. That test used a local worker. It does not establish that a remote peer will beat host RAM on another computer or network.',
      },
    ],
    source: 'https://github.com/amemon12/Hypercompute/blob/main/docs/IMPLEMENTATION_STATUS.md',
  },
];
