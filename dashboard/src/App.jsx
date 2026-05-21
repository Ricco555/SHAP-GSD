import { useState, useEffect } from "react";

const COLORS = {
  bg: "#07090f",
  panel: "#0d1221",
  panelBorder: "#1c2640",
  teal: "#0ee6c0",
  tealDim: "#0a8f78",
  amber: "#f5a40a",
  amberDim: "#8a5d04",
  red: "#e74c3c",
  violet: "#a78bfa",
  blue: "#3b82f6",
  text: "#e4eaf8",
  textDim: "#7a8aaa",
  textMuted: "#3d4e6a",
  node: "#1a3a5c",
  nodeBorder: "#2a5a8c",
  attackNode: "#4a1a2a",
  attackBorder: "#cc3355",
  newNode: "#1a3a1a",
  newBorder: "#22c55e",
};

// Label index → class name
const CLASS_NAMES = ["Benign","Generic","Exploits","Fuzzers","DoS","Recon","Analysis","Backdoor","Shellcode","Worms"];

// All 9 canonical case studies ordered by |top_fg_phi| descending
const CASE_STUDIES = [
  {
    attackClass: "Shellcode", eid: 2117054,
    predClass: "Shellcode", trueClass: "Shellcode", isCorrect: true,
    proba: 0.992,
    probaVec: [0.0, 0.0, 0.005, 0.0, 0.002, 0.0, 0.0, 0.001, 0.992, 0.0],
    features: [
      { name: "DST_PORT_GROUP",          phi:  1.9522 },
      { name: "TCP_WIN_MAX_IN",           phi:  1.3072 },
      { name: "PROTOCOL",                phi:  1.2661 },
      { name: "RETRANSMITTED_IN_BYTES",  phi: -0.7349 },
      { name: "SERVER_TCP_FLAGS",        phi: -0.6070 },
      { name: "SRC_TO_DST_IAT_MAX",      phi:  0.5394 },
    ],
    nInWindow: 26, nTotalNbrs: 25, topTemporalShap: 0.0226,
    temporalNbrs: [
      { eid: 2115550, dt: "t−1.1s", phi: -0.0226 },
      { eid: 2115567, dt: "t−0.2s", phi: -0.0050 },
      { eid: 2115571, dt: "t−0.0s", phi:  0.0012 },
    ],
    nodeShap: [0.4922, -0.0722, -0.0092],
    nodeIds: [28, 29, 27],
    phiNFrac: 0.070, sumPhiN: 0.574,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Port-targeted attack — DST_PORT_GROUP dominant (φ=+1.95). Dense in-window burst (26 neighbors) confirms rapid port sweep.",
  },
  {
    attackClass: "Recon", eid: 2232037,
    predClass: "Recon", trueClass: "Recon", isCorrect: true,
    proba: 0.997,
    probaVec: [0.0, 0.0, 0.001, 0.0, 0.0, 0.997, 0.0, 0.002, 0.0, 0.0],
    features: [
      { name: "DST_PORT_GROUP",   phi:  1.2958 },
      { name: "MIN_IP_PKT_LEN",   phi: -1.1283 },
      { name: "L7_PROTO",         phi:  0.9852 },
      { name: "ICMP_IPV4_TYPE",   phi:  0.6236 },
      { name: "ICMP_TYPE",        phi:  0.5836 },
      { name: "SERVER_TCP_FLAGS", phi:  0.2916 },
    ],
    nInWindow: 0, nTotalNbrs: 0, topTemporalShap: 0.0,
    temporalNbrs: [],
    nodeShap: [0.546, 0.0928, 0.1051, -0.1357, -0.1982, -0.0529, -0.0426, -0.0411, -0.019],
    nodeIds: [30, 29, 28, 13, 17, 16, 14, 9, 18],
    phiNFrac: 0.182, sumPhiN: 1.23,
    srcNovelty: 0.0, dstNovelty: 0.0028,
    narrative: "DNS/ICMP sweep — port targeting + protocol type + 9-node subgraph. No in-window neighbors (median gap 86 min in UNSW-NB15).",
  },
  {
    attackClass: "Backdoor", eid: 2174974,
    predClass: "Backdoor", trueClass: "Backdoor", isCorrect: true,
    proba: 0.916,
    probaVec: [0.0, 0.0, 0.038, 0.0, 0.036, 0.0, 0.0, 0.916, 0.01, 0.0],
    features: [
      { name: "SHORTEST_FLOW_PKT",      phi: 0.9987 },
      { name: "DST_PORT_GROUP",         phi: 0.9511 },
      { name: "RETRANSMITTED_IN_BYTES", phi: 0.8364 },
      { name: "L7_PROTO",              phi: 0.7136 },
      { name: "DST_TO_SRC_IAT_MAX",    phi: 0.3779 },
      { name: "LONGEST_FLOW_PKT",      phi: 0.3522 },
    ],
    nInWindow: 0, nTotalNbrs: 0, topTemporalShap: 0.0,
    temporalNbrs: [],
    nodeShap: [0.007, 0.0895, 0.0131, 0.0434, 0.0329],
    nodeIds: [30, 28, 9, 27, 13],
    phiNFrac: 0.033, sumPhiN: 0.186,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Covert-channel keepalive — short packet size + port + retransmission pattern. No temporal context; isolated long-gap flows.",
  },
  {
    attackClass: "Worms", eid: 2310720,
    predClass: "Worms", trueClass: "Worms", isCorrect: true,
    proba: 0.661,
    probaVec: [0.0, 0.001, 0.237, 0.0, 0.052, 0.046, 0.0, 0.003, 0.0, 0.661],
    features: [
      { name: "ICMP_TYPE",                 phi:  0.7825 },
      { name: "ICMP_IPV4_TYPE",            phi:  0.7295 },
      { name: "SRC_TO_DST_IAT_MAX",        phi: -0.5079 },
      { name: "DST_PORT_GROUP",            phi:  0.4702 },
      { name: "RETRANSMITTED_OUT_PKTS",    phi: -0.4436 },
      { name: "NUM_PKTS_128_TO_256_BYTES", phi:  0.2149 },
    ],
    nInWindow: 0, nTotalNbrs: 0, topTemporalShap: 0.0,
    temporalNbrs: [],
    nodeShap: [0.6609, 0.1727, 0.3467, -1.0576, -0.8879, -0.4546, -0.1852],
    nodeIds: [28, 29, 30, 9, 13, 17, 18],
    phiNFrac: 0.498, sumPhiN: 3.77,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "ICMP scanning — highest φ_N fraction (49.8%). Mixed-sign node weights: model identifies Worms partly despite benign-like neighbour nodes.",
  },
  {
    attackClass: "Generic", eid: 2247735,
    predClass: "Generic", trueClass: "Generic", isCorrect: true,
    proba: 0.918,
    probaVec: [0.0, 0.918, 0.066, 0.001, 0.01, 0.003, 0.0, 0.002, 0.0, 0.0],
    features: [
      { name: "DNS_QUERY_TYPE", phi:  0.6178 },
      { name: "MIN_IP_PKT_LEN", phi:  0.524  },
      { name: "PROTOCOL",       phi: -0.3682 },
      { name: "L7_PROTO",       phi:  0.2541 },
      { name: "DST_PORT_GROUP", phi:  0.2311 },
      { name: "TCP_FLAGS",      phi:  0.068  },
    ],
    nInWindow: 0, nTotalNbrs: 0, topTemporalShap: 0.0,
    temporalNbrs: [],
    nodeShap: [0.5203, 0.4855, 0.0754, -0.0461, -0.1255, -0.3081],
    nodeIds: [30, 29, 28, 13, 17, 18],
    phiNFrac: 0.408, sumPhiN: 1.56,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "DNS-based malware — node novelty accounts for 40.8% of |φ|. Six subgraph nodes drive nearly half the explanatory weight.",
  },
  {
    attackClass: "DoS", eid: 2117597,
    predClass: "DoS", trueClass: "DoS", isCorrect: true,
    proba: 0.762,
    probaVec: [0.009, 0.011, 0.15, 0.0, 0.762, 0.003, 0.002, 0.06, 0.001, 0.001],
    features: [
      { name: "MIN_TTL",                    phi:  0.5893 },
      { name: "NUM_PKTS_512_TO_1024_BYTES", phi:  0.4115 },
      { name: "LONGEST_FLOW_PKT",           phi: -0.3039 },
      { name: "MAX_TTL",                    phi:  0.2988 },
      { name: "DST_PORT_GROUP",             phi: -0.2653 },
      { name: "DURATION_IN",               phi:  0.1452 },
    ],
    nInWindow: 13, nTotalNbrs: 13, topTemporalShap: 0.0022,
    temporalNbrs: [
      { eid: 2115556, dt: "t−0.2s", phi: -0.0022 },
      { eid: 2115550, dt: "t−0.5s", phi: -0.0019 },
    ],
    nodeShap: [0.0501, 0.1526, 0.0537, 0.0608],
    nodeIds: [29, 27, 28, 13],
    phiNFrac: 0.117, sumPhiN: 0.317,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Low TTL + packet-size signature typical of DoS flooding. 13 in-window neighbors confirm traffic burst.",
  },
  {
    attackClass: "Fuzzers", eid: 2117155,
    predClass: "Fuzzers", trueClass: "Fuzzers", isCorrect: true,
    proba: 0.597,
    probaVec: [0.001, 0.032, 0.14, 0.597, 0.067, 0.097, 0.0, 0.01, 0.054, 0.0],
    features: [
      { name: "SRC_TO_DST_IAT_MAX",     phi: -0.3797 },
      { name: "SHORTEST_FLOW_PKT",      phi:  0.3721 },
      { name: "TCP_WIN_MAX_IN",          phi: -0.3432 },
      { name: "DST_TO_SRC_IAT_MAX",     phi: -0.298  },
      { name: "RETRANSMITTED_OUT_PKTS", phi: -0.2448 },
      { name: "DST_PORT_GROUP",         phi:  0.2316 },
    ],
    nInWindow: 28, nTotalNbrs: 28, topTemporalShap: 0.0084,
    temporalNbrs: [
      { eid: 2115567, dt: "t−0.3s", phi:  0.0084 },
      { eid: 2115550, dt: "t−1.2s", phi:  0.0077 },
      { eid: 2115571, dt: "t−0.1s", phi: -0.0002 },
    ],
    nodeShap: [0.0017, 0.447, 0.0028, 0.0019],
    nodeIds: [30, 13, 27, 29],
    phiNFrac: 0.150, sumPhiN: 0.453,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Lowest confidence (0.597) — top feature is negative, suppressing the Fuzzers prediction. Most in-window neighbors of all cases (28).",
  },
  {
    attackClass: "Exploits", eid: 2117278,
    predClass: "DoS", trueClass: "Exploits", isCorrect: false,
    proba: 0.646,
    probaVec: [0.004, 0.033, 0.259, 0.001, 0.646, 0.001, 0.051, 0.003, 0.001, 0.0],
    features: [
      { name: "DST_PORT_GROUP",              phi: -0.2853 },
      { name: "MIN_TTL",                     phi:  0.2717 },
      { name: "RETRANSMITTED_OUT_BYTES",     phi:  0.1412 },
      { name: "MAX_TTL",                     phi:  0.1312 },
      { name: "NUM_PKTS_512_TO_1024_BYTES",  phi: -0.0840 },
      { name: "NUM_PKTS_1024_TO_1514_BYTES", phi:  0.0791 },
    ],
    nInWindow: 25, nTotalNbrs: 25, topTemporalShap: 0.0084,
    temporalNbrs: [
      { eid: 2115556, dt: "t−0.8s", phi: -0.0084 },
    ],
    nodeShap: [-0.0093],
    nodeIds: [29],
    phiNFrac: 0.008, sumPhiN: 0.009,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Misclassification: true Exploits, predicted DoS. TTL pattern overlaps with DoS; DST_PORT_GROUP negatively suppresses the true class.",
  },
  {
    attackClass: "Analysis", eid: 2117614,
    predClass: "DoS", trueClass: "Analysis", isCorrect: false,
    proba: 0.638,
    probaVec: [0.015, 0.024, 0.223, 0.004, 0.638, 0.003, 0.089, 0.003, 0.001, 0.0],
    features: [
      { name: "SRC_PORT_IS_EPHEMERAL",      phi:  0.1241 },
      { name: "SRC_TO_DST_AVG_THROUGHPUT",  phi: -0.0634 },
      { name: "NUM_PKTS_256_TO_512_BYTES",  phi: -0.0595 },
      { name: "NUM_PKTS_512_TO_1024_BYTES", phi: -0.0385 },
      { name: "ICMP_IPV4_TYPE",             phi: -0.0317 },
      { name: "DST_TO_SRC_AVG_THROUGHPUT",  phi: -0.0203 },
    ],
    nInWindow: 25, nTotalNbrs: 25, topTemporalShap: 0.001,
    temporalNbrs: [
      { eid: 2115556, dt: "t−0.8s", phi: -0.0010 },
    ],
    nodeShap: [-0.0084],
    nodeIds: [29],
    phiNFrac: 0.021, sumPhiN: 0.008,
    srcNovelty: 0.0, dstNovelty: 0.0,
    narrative: "Misclassification: true Analysis, predicted DoS. Weakest feature signal (top φ=0.12); consistent Analysis/DoS confusion in overall metrics.",
  },
];

// Derived constants — computed once from CASE_STUDIES data
const MAX_NODES   = Math.max(...CASE_STUDIES.map(c => c.nodeIds.length));   // 9 (Recon)
const MAX_T_ROWS  = 3;  // always render this many temporal rows

// Per-class attack signature text shown in the bottom bar
const CASE_SIGNATURES = {
  Shellcode: "DST_PORT_GROUP (φ=+1.95) — specific destination port identifies the shellcode delivery channel.",
  Recon:     "DST_PORT_GROUP + ICMP sweep (φ=+1.30) — multi-protocol port scan across a 9-node subgraph.",
  Backdoor:  "SHORTEST_FLOW_PKT (φ=+1.00) — keepalive-sized packets signal a covert channel.",
  Worms:     "ICMP_TYPE + ICMP_IPV4_TYPE (φ=+0.78) — broadcast scanning sweep; 49.8% attributed to node subgraph.",
  Generic:   "DNS_QUERY_TYPE (φ=+0.62) — DNS-tunnelled traffic; node subgraph accounts for 40.8% of |φ|.",
  DoS:       "MIN_TTL (φ=+0.59) — low hop-count typical of flooding; confirmed by 13 in-window burst neighbours.",
  Fuzzers:   "SRC_TO_DST_IAT_MAX (φ=−0.38) — irregular inter-arrival suppresses prediction; model uncertain (0.597).",
  Exploits:  "MIN_TTL overlaps DoS signature (φ=+0.27) — feature ambiguity drives misclassification to DoS.",
  Analysis:  "SRC_PORT_IS_EPHEMERAL (φ=+0.12) — weakest signal in the set; consistent confusion with DoS in metrics.",
};

function featureColor(phi) {
  if (phi < 0) return COLORS.violet;
  if (phi >= 1.0) return COLORS.red;
  if (phi >= 0.4) return COLORS.amber;
  return COLORS.teal;
}

// ─── tiny animated network ──────────────────────────────────────────────────
function MiniGraph({ width, height }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 120), 60);
    return () => clearInterval(id);
  }, []);

  const nodes = [
    { id: 0, x: 0.22, y: 0.50, label: "10.0.0.5",  type: "attacker" },
    { id: 1, x: 0.55, y: 0.25, label: "10.0.0.12", type: "victim" },
    { id: 2, x: 0.55, y: 0.50, label: "10.0.0.7",  type: "victim" },
    { id: 3, x: 0.55, y: 0.75, label: "10.0.0.19", type: "victim" },
    { id: 4, x: 0.82, y: 0.38, label: "10.0.0.3",  type: "victim" },
    { id: 5, x: 0.82, y: 0.62, label: "172.x.x.1", type: "new" },
  ];

  const edges = [
    { s: 0, d: 1, t: 10, w: 1.0, highlight: true },
    { s: 0, d: 2, t: 24, w: 0.8, highlight: true },
    { s: 0, d: 3, t: 38, w: 0.9, highlight: true },
    { s: 1, d: 4, t: 50, w: 0.5, highlight: false },
    { s: 2, d: 5, t: 62, w: 0.7, highlight: false },
  ];

  const animProgress = (tick % 120) / 120;

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke={COLORS.textMuted} strokeWidth="0.3" opacity="0.3" />
        </pattern>
        <marker id="arrow-r" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={COLORS.red} opacity="0.8" />
        </marker>
        <marker id="arrow-g" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={COLORS.teal} opacity="0.5" />
        </marker>
      </defs>
      <rect width={width} height={height} fill="url(#grid)" rx="8" />

      {edges.map((e, i) => {
        const s = nodes[e.s], d = nodes[e.d];
        const sx = s.x * width, sy = s.y * height;
        const dx = d.x * width, dy = d.y * height;
        const len = Math.hypot(dx - sx, dy - sy);
        const ux = (dx - sx) / len, uy = (dy - sy) / len;
        const ex = dx - ux * 9, ey = dy - uy * 9;
        const pulse = (animProgress * 1.4 - i * 0.22 + 2) % 1;
        const px = sx + (ex - sx) * pulse, py = sy + (ey - sy) * pulse;
        const visible = pulse > 0 && pulse < 1;
        return (
          <g key={i}>
            <line x1={sx} y1={sy} x2={ex} y2={ey}
              stroke={e.highlight ? COLORS.red : COLORS.teal}
              strokeWidth={e.highlight ? 1.5 : 1}
              opacity={e.highlight ? 0.7 : 0.3}
              markerEnd={e.highlight ? "url(#arrow-r)" : "url(#arrow-g)"}
            />
            <text x={(sx + ex) / 2} y={(sy + ey) / 2 - 5}
              fill={COLORS.textMuted} fontSize="9" textAnchor="middle">
              t={e.t}s
            </text>
            {visible && (
              <circle cx={px} cy={py} r="3"
                fill={e.highlight ? COLORS.red : COLORS.teal} opacity="0.9" />
            )}
          </g>
        );
      })}

      {nodes.map(n => {
        const cx = n.x * width, cy = n.y * height;
        const fill = n.type === "attacker" ? COLORS.attackNode
          : n.type === "new" ? COLORS.newNode : COLORS.node;
        const stroke = n.type === "attacker" ? COLORS.attackBorder
          : n.type === "new" ? COLORS.newBorder : COLORS.nodeBorder;
        return (
          <g key={n.id}>
            <circle cx={cx} cy={cy} r={n.type === "attacker" ? 12 : 9}
              fill={fill} stroke={stroke} strokeWidth="1.5" />
            {n.type === "new" && (
              <text x={cx + 11} y={cy - 7} fill={COLORS.newBorder}
                fontSize="9" fontWeight="bold">NEW</text>
            )}
            <text x={cx} y={cy + 18}
              fill={n.type === "attacker" ? COLORS.red : COLORS.textDim}
              fontSize="9" textAnchor="middle">
              {n.label.split(".").slice(-2).join(".")}
            </text>
          </g>
        );
      })}

      <text x={nodes[0].x * width} y={nodes[0].y * height - 18}
        fill={COLORS.red} fontSize="11" textAnchor="middle" fontWeight="bold">
        scanner
      </text>
    </svg>
  );
}

// ─── coalition z-vector illustration ────────────────────────────────────────
function CoalitionBox({ label, color, children, glyph }) {
  return (
    <div style={{
      border: `1px solid ${color}40`,
      borderLeft: `3px solid ${color}`,
      borderRadius: "6px",
      padding: "7px 10px",
      background: `${color}08`,
      display: "flex",
      alignItems: "center",
      gap: "8px",
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: `${color}22`, border: `1px solid ${color}60`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 15, flexShrink: 0,
      }}>{glyph}</div>
      <div style={{ flex: 1 }}>
        <div style={{ color, fontSize: 12, fontWeight: 700, letterSpacing: 1, marginBottom: 2 }}>
          {label}
        </div>
        <div style={{ color: COLORS.textDim, fontSize: 11 }}>{children}</div>
      </div>
    </div>
  );
}

// ─── attribution bar chart ───────────────────────────────────────────────────
function FeatureBar({ name, phi, color, maxPhi }) {
  const pct = Math.abs(phi) / maxPhi;
  const neg = phi < 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
      <div style={{ width: 140, textAlign: "right", fontSize: 10, color: COLORS.textDim,
        fontFamily: "monospace", flexShrink: 0 }}>{name}</div>
      <div style={{ flex: 1, position: "relative", height: 13, background: COLORS.textMuted + "20",
        borderRadius: 2 }}>
        <div style={{
          position: "absolute",
          [neg ? "right" : "left"]: "50%",
          width: `${pct * 50}%`,
          height: "100%",
          background: neg ? COLORS.violet : color,
          borderRadius: 2,
          opacity: 0.85,
        }} />
        <div style={{
          position: "absolute", left: "50%", top: 0, width: 1,
          height: "100%", background: COLORS.textMuted, opacity: 0.5,
        }} />
      </div>
      <div style={{ width: 42, fontSize: 11, color, fontFamily: "monospace",
        textAlign: "right", flexShrink: 0 }}>
        {phi > 0 ? "+" : ""}{phi.toFixed(2)}
      </div>
    </div>
  );
}

// ─── temporal neighbors panel ────────────────────────────────────────────────
function TemporalNeighborRow({ eid, phi, dt }) {
  const pos = phi >= 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 3 }}>
      <div style={{
        width: 20, height: 20, borderRadius: 3,
        background: `${COLORS.amber}20`,
        border: `1px solid ${COLORS.amber}50`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 9, color: COLORS.amber, flexShrink: 0,
      }}>→</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 11, color: COLORS.text, fontFamily: "monospace" }}>
          edge {eid}
        </div>
        <div style={{ fontSize: 9, color: COLORS.textMuted }}>{dt}</div>
      </div>
      <div style={{ fontSize: 11, color: pos ? COLORS.amber : COLORS.violet, fontFamily: "monospace" }}>
        {phi > 0 ? "+" : ""}{phi.toFixed(4)}
      </div>
    </div>
  );
}

// ─── φ_N fraction bar ────────────────────────────────────────────────────────
function PhiNBar({ frac }) {
  const pct = frac * 100;
  const barColor = pct > 35 ? COLORS.violet : pct > 15 ? COLORS.amber : COLORS.tealDim;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: COLORS.textDim }}>φ_N share of total |φ|</span>
        <span style={{ fontSize: 11, fontFamily: "monospace", fontWeight: 700, color: barColor }}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 6, background: COLORS.textMuted + "25", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: `linear-gradient(90deg, ${barColor}99, ${barColor})`,
          borderRadius: 3,
          transition: "width 0.35s ease",
        }} />
      </div>
    </div>
  );
}

// ─── node shap rows ──────────────────────────────────────────────────────────
function NodeShapRow({ nodeId, phi }) {
  const pos = phi >= 0;
  const color = pos ? COLORS.violet : COLORS.textDim;
  const maxAbs = 1.2;
  const pct = Math.min(Math.abs(phi) / maxAbs * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
      <div style={{
        width: 30, fontSize: 9, fontFamily: "monospace", textAlign: "right",
        color: COLORS.textMuted, flexShrink: 0,
      }}>N{nodeId}</div>
      <div style={{ flex: 1, position: "relative", height: 8, background: COLORS.textMuted + "20", borderRadius: 2 }}>
        <div style={{
          position: "absolute",
          [pos ? "left" : "right"]: "50%",
          width: `${pct * 0.5}%`,
          height: "100%",
          background: pos ? COLORS.violet : COLORS.textDim,
          borderRadius: 2, opacity: 0.75,
        }} />
        <div style={{
          position: "absolute", left: "50%", top: 0, width: 1,
          height: "100%", background: COLORS.textMuted, opacity: 0.4,
        }} />
      </div>
      <div style={{ width: 44, fontSize: 10, fontFamily: "monospace", textAlign: "right",
        color, flexShrink: 0 }}>
        {phi > 0 ? "+" : ""}{phi.toFixed(3)}
      </div>
    </div>
  );
}

// ─── nav button ──────────────────────────────────────────────────────────────
function NavBtn({ onClick, children, style: extraStyle }) {
  return (
    <button onClick={onClick} style={{
      background: COLORS.panel,
      border: `1px solid ${COLORS.panelBorder}`,
      borderRadius: 4,
      color: COLORS.textDim,
      fontSize: 14,
      padding: "2px 8px",
      cursor: "pointer",
      lineHeight: 1.4,
      flexShrink: 0,
      transition: "border-color 0.15s, color 0.15s",
      ...extraStyle,
    }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = COLORS.teal; e.currentTarget.style.color = COLORS.teal; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = COLORS.panelBorder; e.currentTarget.style.color = COLORS.textDim; }}
    >{children}</button>
  );
}

// ─── main ────────────────────────────────────────────────────────────────────
export default function GraphicalAbstract() {
  const [caseIdx, setCaseIdx] = useState(0);

  const cs = CASE_STUDIES[caseIdx];
  const maxPhi = Math.max(...cs.features.map(f => Math.abs(f.phi)));
  const badgeColor = cs.isCorrect ? COLORS.red : COLORS.amber;

  const prev = () => setCaseIdx(i => (i - 1 + CASE_STUDIES.length) % CASE_STUDIES.length);
  const next = () => setCaseIdx(i => (i + 1) % CASE_STUDIES.length);

  return (
    <div style={{
      width: "100%", maxWidth: 1020, margin: "0 auto",
      background: COLORS.bg,
      fontFamily: "'DM Mono', 'Fira Code', monospace",
      padding: "20px 20px 24px",
      borderRadius: 12,
      boxSizing: "border-box",
    }}>
      {/* ── header ── */}
      <div style={{ textAlign: "center", marginBottom: 18 }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 12,
          background: COLORS.panel, border: `1px solid ${COLORS.panelBorder}`,
          borderRadius: 8, padding: "8px 20px",
        }}>
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.teal}, ${COLORS.amber})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            fontSize: 35, fontWeight: 900, letterSpacing: -1,
          }}>SHAP-GSD</div>
          <div style={{ width: 1, height: 28, background: COLORS.panelBorder }} />
          <div style={{ textAlign: "left" }}>
            <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600 }}>
              Multi-Granularity Shapley Explanations
            </div>
            <div style={{ color: COLORS.textDim, fontSize: 12 }}>
              for Graph Neural Network Intrusion Detection
            </div>
          </div>
        </div>
      </div>

      {/* ── three-column layout ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 2px 1.05fr 2px 1.15fr", gap: "0 0" }}>

        {/* ══ COL 1: Input ══ */}
        <div style={{ paddingRight: 14 }}>
          <SectionLabel color={COLORS.teal}>① INPUT</SectionLabel>
          <PanelBox>
            <div style={{ fontSize: 12, color: COLORS.textDim, marginBottom: 8 }}>
              NetFlow stream → IP-level temporal graph
            </div>
            <MiniGraph width={200} height={130} />
            <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 4 }}>
              <LegendItem color={COLORS.attackBorder} label="Source (scanner)" shape="circle" />
              <LegendItem color={COLORS.nodeBorder}   label="Destination host" shape="circle" />
              <LegendItem color={COLORS.newBorder}    label="Novel / unseen IP" shape="circle" />
              <LegendItem color={COLORS.red}          label="Suspicious flow (labeled)" shape="line" />
            </div>
            <div style={{
              marginTop: 10, padding: "6px 8px",
              background: `${COLORS.teal}10`, borderRadius: 4,
              border: `1px solid ${COLORS.teal}30`,
            }}>
              <div style={{ fontSize: 11, color: COLORS.teal, fontWeight: 700, marginBottom: 3 }}>
                SHAP-GSD model
              </div>
              {[
                "IP-level nodes (host graph)",
                "15-dim temporal node state",
                "Temporally-constrained sampling",
                "d_e = 218, K = 48 feature groups",
              ].map((t, i) => (
                <div key={i} style={{ fontSize: 10, color: COLORS.textDim, display: "flex", gap: 4 }}>
                  <span style={{ color: COLORS.teal }}>▸</span> {t}
                </div>
              ))}
            </div>
          </PanelBox>
        </div>

        <Divider />

        {/* ══ COL 2: Method ══ */}
        <div style={{ paddingLeft: 14, paddingRight: 14 }}>
          <SectionLabel color={COLORS.amber}>② SHAP-GSD METHOD</SectionLabel>
          <PanelBox>
            <div style={{ fontSize: 12, color: COLORS.textDim, marginBottom: 10 }}>
              Unified coalition vector over three granularities:
            </div>

            <div style={{
              textAlign: "center", padding: "8px", marginBottom: 10,
              background: `${COLORS.amber}08`, border: `1px solid ${COLORS.amber}25`,
              borderRadius: 6, fontSize: 15,
            }}>
              <span style={{ color: COLORS.amber, fontWeight: 700 }}>z</span>
              <span style={{ color: COLORS.textDim }}> = [ </span>
              <span style={{ color: COLORS.red }}>z_F</span>
              <span style={{ color: COLORS.textDim }}> | </span>
              <span style={{ color: COLORS.amber }}>z_T</span>
              <span style={{ color: COLORS.textDim }}> | </span>
              <span style={{ color: COLORS.violet }}>z_N</span>
              <span style={{ color: COLORS.textDim }}> ]</span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <CoalitionBox label="φ_F — Semantic Feature Groups" color={COLORS.red} glyph="F">
                48 groups (39 numeric + 9 categorical one-hots). KernelSHAP over
                variable-level partition.
              </CoalitionBox>
              <CoalitionBox label="φ_T — Temporal Neighborhood" color={COLORS.amber} glyph="T">
                Mask neighbor edge e′ at t′ ≤ t_e: roll back node state to pre-e′.
                Answers "did this prior flow cause the alert?"
              </CoalitionBox>
              <CoalitionBox label="φ_N — Node Novelty / State" color={COLORS.violet} glyph="N">
                Novelty flag zeroed → treats node as known. Quantifies "did a
                previously-unseen IP drive this classification?"
              </CoalitionBox>
            </div>

            <div style={{ marginTop: 10, display: "flex", gap: 4, flexWrap: "wrap" }}>
              {["Efficiency", "Dummy", "Symmetry", "Additivity"].map(a => (
                <div key={a} style={{
                  fontSize: 9, padding: "2px 6px", borderRadius: 3,
                  background: `${COLORS.teal}15`, border: `1px solid ${COLORS.teal}35`,
                  color: COLORS.teal,
                }}>✓ {a}</div>
              ))}
            </div>

            <div style={{
              marginTop: 10, display: "flex", alignItems: "center", gap: 4,
              justifyContent: "center",
            }}>
              {["M=512 coalitions", "→", "Weighted linear surrogate", "→", "Shapley φ"].map((s, i) => (
                <div key={i} style={{
                  fontSize: s === "→" ? 16 : 10,
                  color: s === "→" ? COLORS.amber : COLORS.textDim,
                }}>{s}</div>
              ))}
            </div>

            <div style={{
              marginTop: 8, padding: "5px 8px",
              background: `${COLORS.violet}10`, border: `1px solid ${COLORS.violet}25`,
              borderRadius: 4, fontSize: 10, color: COLORS.textDim,
            }}>
              <span style={{ color: COLORS.violet }}>◆ </span>
              Class-conditional backgrounds from training split only — no temporal leakage in explanations.
            </div>
          </PanelBox>
        </div>

        <Divider />

        {/* ══ COL 3: Output — dynamic ══ */}
        <div style={{ paddingLeft: 14 }}>
          <SectionLabel color={COLORS.violet}>③ MULTI-GRANULARITY EXPLANATION</SectionLabel>
          <PanelBox>

            {/* class chip strip — row 1: 5 pills, row 2: 4 pills, each filling full width */}
            <div style={{ display: "flex", flexDirection: "column", gap: 3, marginBottom: 9 }}>
              {[CASE_STUDIES.slice(0, 5), CASE_STUDIES.slice(5)].map((row, rowIdx) => (
                <div key={rowIdx} style={{
                  display: "grid",
                  gridTemplateColumns: `repeat(${row.length}, 1fr)`,
                  gap: 3,
                }}>
                  {row.map((c, colIdx) => {
                    const i = rowIdx === 0 ? colIdx : colIdx + 5;
                    const active = i === caseIdx;
                    const chipColor = c.isCorrect ? COLORS.red : COLORS.amber;
                    return (
                      <button key={i} onClick={() => setCaseIdx(i)} style={{
                        padding: "4px 2px", borderRadius: 3, cursor: "pointer",
                        fontSize: 10, fontFamily: "monospace", fontWeight: 700,
                        letterSpacing: 0.2, textAlign: "center",
                        background: active ? `${chipColor}20` : "transparent",
                        border: `1px solid ${active ? chipColor : COLORS.textMuted}`,
                        color: active ? chipColor : COLORS.textMuted,
                        outline: "none",
                        transition: "border-color 0.15s, color 0.15s, background 0.15s",
                      }}>
                        {c.attackClass}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* case header: prev | attack card | confidence card | next */}
            <div style={{ display: "flex", alignItems: "stretch", gap: 6, marginBottom: 8 }}>
              <NavBtn onClick={prev} style={{ alignSelf: "center" }}>◀</NavBtn>

              {/* attack class card — mirrors the confidence card */}
              <div style={{
                flex: 1, minWidth: 0,
                padding: "4px 10px",
                background: `${badgeColor}14`,
                border: `1px solid ${badgeColor}50`,
                borderRadius: 5,
                display: "flex", flexDirection: "column", justifyContent: "center",
              }}>
                <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 0.5, marginBottom: 1 }}>
                  attack class
                </div>
                <div style={{ fontSize: 16, fontWeight: 700, color: badgeColor, lineHeight: 1.2 }}>
                  {cs.isCorrect ? "⚠" : "✕"} {cs.attackClass}
                </div>
                {!cs.isCorrect && (
                  <div style={{ fontSize: 9, color: COLORS.amber, marginTop: 2 }}>
                    → pred {cs.predClass}
                  </div>
                )}
              </div>

              {/* confidence card */}
              <div style={{
                flexShrink: 0,
                padding: "4px 10px",
                background: COLORS.panel,
                border: `1px solid ${COLORS.teal}40`,
                borderRadius: 5,
                textAlign: "center",
                minWidth: 72,
                display: "flex", flexDirection: "column", justifyContent: "center",
              }}>
                <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 0.5, marginBottom: 1 }}>
                  confidence
                </div>
                <div style={{
                  fontSize: 20, fontWeight: 900,
                  fontFamily: "'DM Mono', 'Fira Code', monospace",
                  color: COLORS.text, lineHeight: 1.1,
                }}>
                  {cs.proba.toFixed(3)}
                </div>
              </div>

              <NavBtn onClick={next} style={{ alignSelf: "center" }}>▶</NavBtn>
            </div>

            {/* φ_F panel */}
            <SubPanel label="φ_F  Feature-Group Attribution" color={COLORS.red}>
              {cs.features.map((f, i) => (
                <FeatureBar key={i} name={f.name} phi={f.phi}
                  color={featureColor(f.phi)} maxPhi={maxPhi} />
              ))}
              <div style={{
                marginTop: 4, fontSize: 10, color: COLORS.textDim,
                borderTop: `1px solid ${COLORS.panelBorder}`, paddingTop: 4,
              }}>
                {cs.narrative}
              </div>
            </SubPanel>

            {/* φ_T panel — always renders MAX_T_ROWS rows + 1 stat strip for stable height */}
            <SubPanel label="φ_T  Temporal Neighborhood" color={COLORS.amber}>
              {Array.from({ length: MAX_T_ROWS }, (_, i) => {
                const n = cs.temporalNbrs[i];
                return n
                  ? <TemporalNeighborRow key={i} eid={n.eid} phi={n.phi} dt={n.dt} />
                  : <div key={i} style={{ height: 29, marginBottom: 3, visibility: "hidden" }} />;
              })}
              {/* stat/message strip — same padding/structure regardless of case */}
              {cs.temporalNbrs.length > 0 ? (
                <div style={{
                  marginTop: 4, padding: "4px 6px",
                  background: `${COLORS.amber}10`, borderRadius: 3,
                  fontSize: 10, color: COLORS.amberDim,
                }}>
                  {cs.nInWindow} in-window / {cs.nTotalNbrs} total neighbors
                  {" · "}max |φ_T| = {cs.topTemporalShap.toFixed(4)}
                </div>
              ) : (
                <div style={{
                  padding: "4px 6px",
                  background: `${COLORS.textMuted}10`, borderRadius: 3,
                  fontSize: 10, color: COLORS.textMuted,
                }}>
                  No in-window neighbors (W=60s) — temporal attribution = 0.
                  Long-gap flows typical of UNSW-NB15 (median Δt ≈ 86 min).
                </div>
              )}
            </SubPanel>

            {/* φ_N panel — always renders MAX_NODES rows for stable height */}
            <SubPanel label="φ_N  Node Novelty & State" color={COLORS.violet}>
              <PhiNBar frac={cs.phiNFrac} />
              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {Array.from({ length: MAX_NODES }, (_, i) => {
                  const nodeId = cs.nodeIds[i];
                  const phi    = cs.nodeShap[i];
                  return nodeId !== undefined
                    ? <NodeShapRow key={i} nodeId={nodeId} phi={phi} />
                    : <div key={i} style={{ height: 10, marginBottom: 2, visibility: "hidden" }} />;
                })}
              </div>
              {/* novelty strip — always reserves space; invisible when both are zero */}
              <div style={{
                marginTop: 5, fontSize: 10,
                color: (cs.srcNovelty !== 0 || cs.dstNovelty !== 0) ? COLORS.newBorder : "transparent",
                background: (cs.srcNovelty !== 0 || cs.dstNovelty !== 0) ? `${COLORS.newBorder}10` : "transparent",
                borderRadius: 3, padding: "3px 6px",
                border: (cs.srcNovelty !== 0 || cs.dstNovelty !== 0) ? "none" : "1px solid transparent",
              }}>
                ★ novelty — src: {cs.srcNovelty.toFixed(4)}  dst: {cs.dstNovelty.toFixed(4)}
              </div>
              <div style={{ marginTop: 5, fontSize: 10, color: COLORS.textDim }}>
                sum |φ_N| = {cs.sumPhiN.toFixed(3)} across {cs.nodeIds.length} node{cs.nodeIds.length !== 1 ? "s" : ""}
              </div>
            </SubPanel>

          </PanelBox>
        </div>
      </div>

      {/* ── bottom bar: key findings ── */}
      {/* ── bottom bar: first two are method constants; last two update per case ── */}
      {(() => {
        const pct = (cs.phiNFrac * 100).toFixed(1);
        const n   = cs.nodeIds.length;
        const nodeQualifier = cs.phiNFrac > 0.35
          ? "node subgraph drives the classification"
          : cs.phiNFrac > 0.15
          ? "node-state contributes alongside features"
          : "feature signal dominates over node-state";

        const cards = [
          {
            icon: "◈", color: COLORS.red, title: "Semantic Grouping",
            body: "48 groups vs 601 raw dims. Eliminates surrogate ill-conditioning.",
            dynamic: false,
          },
          {
            icon: "⧖", color: COLORS.amber, title: "Temporal Faithfulness",
            body: "Zero-violation temporal sampling. Explanations match deployment semantics.",
            dynamic: false,
          },
          {
            icon: "◎", color: COLORS.violet, title: "Node-Level Attribution",
            body: `φ_N = ${pct}% across ${n} node${n !== 1 ? "s" : ""} — ${nodeQualifier}.`,
            dynamic: true,
          },
          {
            icon: "✦", color: COLORS.teal, title: "Attack Signatures",
            body: CASE_SIGNATURES[cs.attackClass],
            dynamic: true,
          },
        ];

        return (
          <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
            {cards.map((c, i) => (
              <div key={i} style={{
                background: COLORS.panel,
                border: `1px solid ${c.color}30`,
                borderTop: `2px solid ${c.color}`,
                borderRadius: 6, padding: "8px 10px",
                opacity: c.dynamic ? 1 : 0.8,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
                  <span style={{ color: c.color, fontSize: 18 }}>{c.icon}</span>
                  <span style={{ color: c.color, fontSize: 12, fontWeight: 700 }}>{c.title}</span>
                  {c.dynamic && (
                    <span style={{
                      marginLeft: "auto", fontSize: 8, color: c.color,
                      opacity: 0.6, letterSpacing: 0.5,
                    }}>per-case</span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: COLORS.textDim, lineHeight: 1.5 }}>{c.body}</div>
              </div>
            ))}
          </div>
        );
      })()}
    </div>
  );
}

// ─── helpers ─────────────────────────────────────────────────────────────────
function SectionLabel({ color, children }) {
  return (
    <div style={{
      color, fontSize: 11, fontWeight: 700, letterSpacing: 1.5,
      marginBottom: 6, display: "flex", alignItems: "center", gap: 5,
    }}>
      <div style={{ flex: 1, height: 1, background: `${color}40` }} />
      {children}
      <div style={{ flex: 1, height: 1, background: `${color}40` }} />
    </div>
  );
}

function PanelBox({ children }) {
  return (
    <div style={{
      background: COLORS.panel,
      border: `1px solid ${COLORS.panelBorder}`,
      borderRadius: 8,
      padding: "10px 12px",
    }}>{children}</div>
  );
}

function Divider() {
  return (
    <div style={{
      width: 2, background: `linear-gradient(to bottom, transparent, ${COLORS.panelBorder}, transparent)`,
      margin: "0 0",
    }} />
  );
}

function SubPanel({ label, color, children }) {
  return (
    <div style={{
      marginBottom: 7,
      border: `1px solid ${color}25`,
      borderLeft: `2px solid ${color}`,
      borderRadius: "0 5px 5px 0",
      padding: "6px 8px",
      background: `${color}06`,
    }}>
      <div style={{ fontSize: 11, color, fontWeight: 700, marginBottom: 6 }}>{label}</div>
      {children}
    </div>
  );
}

function LegendItem({ color, label, shape }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: COLORS.textDim }}>
      {shape === "circle"
        ? <div style={{ width: 8, height: 8, borderRadius: "50%", border: `1.5px solid ${color}`, background: `${color}20`, flexShrink: 0 }} />
        : <div style={{ width: 12, height: 2, background: color, opacity: 0.7, flexShrink: 0 }} />
      }
      {label}
    </div>
  );
}
