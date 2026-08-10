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

// Label index -> class name, real label_map.json order (Benign=0, Analysis=1, ...)
const CLASS_NAMES = ["Benign","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Recon","Shellcode","Worms"];

// All 8 canonical case studies (curated selection, main.tex subsec:casestudies) ordered by |top_fg_phi| descending.
// Regenerated 2026-08-10 from runs/nf_unsw_nb15_v3_r3_s2/outputs/figures/case_studies/ (armRb+S2 run,
// post l1_reg-truncation fix). Backdoor dropped: no explained Backdoor flow in this run is correctly
// classified and none is predicted Backdoor at all (main.tex subsec:disc_detection) -- the previous
// hardcoded Backdoor entry here predates that finding and is invalid under the current model.
const CASE_STUDIES = [
  {
    attackClass: "Exploits", eid: 1945754,
    predClass: "Exploits", trueClass: "Exploits", isCorrect: true,
    proba: 0.620,
    probaVec: [0.086, 0.000, 0.003, 0.269, 0.620, 0.002, 0.018, 0.001, 0.000, 0.000],
    features: [
      { name: "RETRANSMITTED_IN_PKTS", phi: 1.0910 },
      { name: "LONGEST_FLOW_PKT", phi: 0.5291 },
      { name: "IN_BYTES", phi: 0.4856 },
      { name: "NUM_PKTS_256_TO_512_BYTES", phi: -0.4036 },
      { name: "NUM_PKTS_1024_TO_1514_BYTES", phi: 0.3672 },
      { name: "RETRANSMITTED_IN_BYTES", phi: 0.3288 },
    ],
    nInWindow: 15, nTotalNbrs: 15, topTemporalShap: 0.0330,
    temporalNbrs: [
      { eid: 1945380, dt: "t−12.4s", phi: -0.0330 },
      { eid: 1945342, dt: "t−14.1s", phi: -0.0330 },
      { eid: 1945529, dt: "t−7.7s", phi: -0.0330 },
    ],
    nodeShap: [0.0436, 0.0806, 0.0418],
    nodeIds: [29, 28, 30],
    phiNFrac: 0.030, sumPhiN: 0.166,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "Retransmitted-packet-count signature (φ=+1.09) — the strongest top-feature magnitude in the current case set. 15 in-window neighbours.",
  },
  {
    attackClass: "Recon", eid: 1929497,
    predClass: "Recon", trueClass: "Recon", isCorrect: true,
    proba: 0.994,
    probaVec: [0.000, 0.000, 0.001, 0.002, 0.002, 0.000, 0.000, 0.994, 0.000, 0.000],
    features: [
      { name: "DST_PORT_GROUP", phi: 1.0663 },
      { name: "L7_PROTO", phi: 1.0218 },
      { name: "ICMP_TYPE", phi: 0.4462 },
      { name: "ICMP_IPV4_TYPE", phi: 0.4415 },
      { name: "SHORTEST_FLOW_PKT", phi: 0.3493 },
      { name: "NUM_PKTS_UP_TO_128_BYTES", phi: 0.2228 },
    ],
    nInWindow: 30, nTotalNbrs: 30, topTemporalShap: 0.0270,
    temporalNbrs: [
      { eid: 1928318, dt: "t−44.3s", phi: -0.0270 },
      { eid: 1928319, dt: "t−44.3s", phi: -0.0258 },
      { eid: 1928648, dt: "t−32.9s", phi: -0.0251 },
    ],
    nodeShap: [0.0603, 0.0392, 0.0277, -0.0084],
    nodeIds: [29, 30, 27, 14],
    phiNFrac: 0.023, sumPhiN: 0.136,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "Destination-port sweep (φ=+1.07); 30 in-window neighbours, the most of any case in the set.",
  },
  {
    attackClass: "Shellcode", eid: 1929451,
    predClass: "Shellcode", trueClass: "Shellcode", isCorrect: true,
    proba: 0.704,
    probaVec: [0.002, 0.000, 0.018, 0.085, 0.054, 0.006, 0.031, 0.087, 0.704, 0.013],
    features: [
      { name: "PROTOCOL", phi: 0.8659 },
      { name: "DST_PORT_GROUP", phi: 0.8179 },
      { name: "TCP_WIN_MAX_IN", phi: 0.6919 },
      { name: "MIN_IP_PKT_LEN", phi: -0.6421 },
      { name: "TCP_WIN_MAX_OUT", phi: -0.3286 },
      { name: "L7_PROTO", phi: 0.2825 },
    ],
    nInWindow: 19, nTotalNbrs: 19, topTemporalShap: 0.0107,
    temporalNbrs: [
      { eid: 1928288, dt: "t−43.5s", phi: -0.0107 },
      { eid: 1928298, dt: "t−43.0s", phi: -0.0106 },
      { eid: 1928287, dt: "t−43.5s", phi: -0.0106 },
    ],
    nodeShap: [0.1069, 0.0937, 0.0141, -0.0021],
    nodeIds: [28, 27, 30, 14],
    phiNFrac: 0.033, sumPhiN: 0.217,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "Protocol-field signature (φ=+0.87) identifies the delivery channel; 19 in-window neighbours.",
  },
  {
    attackClass: "Generic", eid: 2250911,
    predClass: "Generic", trueClass: "Generic", isCorrect: true,
    proba: 0.997,
    probaVec: [0.000, 0.000, 0.000, 0.002, 0.000, 0.000, 0.997, 0.001, 0.000, 0.000],
    features: [
      { name: "DNS_QUERY_TYPE", phi: 0.8630 },
      { name: "DNS_QUERY_ID", phi: 0.6033 },
      { name: "DST_PORT_GROUP", phi: 0.4564 },
      { name: "L7_PROTO", phi: 0.4531 },
      { name: "MIN_IP_PKT_LEN", phi: 0.1806 },
      { name: "RETRANSMITTED_OUT_BYTES", phi: -0.1593 },
    ],
    nInWindow: 3, nTotalNbrs: 3, topTemporalShap: 0.0044,
    temporalNbrs: [
      { eid: 2248801, dt: "t−55.8s", phi: 0.0044 },
      { eid: 2249141, dt: "t−48.3s", phi: 0.0044 },
      { eid: 2249737, dt: "t−31.4s", phi: -0.0036 },
    ],
    nodeShap: [0.0837, 0.0442, -0.0370, 0.1354, 0.0037, 0.0037, 0.0040, 0.0037, 0.0303],
    nodeIds: [28, 30, 29, 13, 17, 11, 4, 18, 14],
    phiNFrac: 0.069, sumPhiN: 0.346,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "DNS query-type signature (φ=+0.86); highest-confidence case in the set (99.7%), despite a sparse 3-neighbour window.",
  },
  {
    attackClass: "Fuzzers", eid: 1931492,
    predClass: "Fuzzers", trueClass: "Fuzzers", isCorrect: true,
    proba: 0.832,
    probaVec: [0.000, 0.000, 0.007, 0.041, 0.043, 0.832, 0.005, 0.069, 0.003, 0.000],
    features: [
      { name: "NUM_PKTS_128_TO_256_BYTES", phi: 0.5811 },
      { name: "L7_PROTO", phi: 0.2808 },
      { name: "MIN_IP_PKT_LEN", phi: 0.1252 },
      { name: "RETRANSMITTED_IN_PKTS", phi: 0.1204 },
      { name: "RETRANSMITTED_IN_BYTES", phi: 0.1187 },
      { name: "DST_TO_SRC_IAT_AVG", phi: 0.1068 },
    ],
    nInWindow: 28, nTotalNbrs: 28, topTemporalShap: 0.0074,
    temporalNbrs: [
      { eid: 1930354, dt: "t−45.5s", phi: 0.0074 },
      { eid: 1931297, dt: "t−7.2s", phi: 0.0073 },
      { eid: 1931462, dt: "t−1.4s", phi: 0.0071 },
    ],
    nodeShap: [0.0107, -0.0094, 0.0434],
    nodeIds: [14, 29, 28],
    phiNFrac: 0.023, sumPhiN: 0.063,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "Packet-size-bucket signature (φ=+0.58); 28 in-window neighbours.",
  },
  {
    attackClass: "Worms", eid: 2310720,
    predClass: "Worms", trueClass: "Worms", isCorrect: true,
    proba: 0.595,
    probaVec: [0.001, 0.001, 0.005, 0.062, 0.160, 0.004, 0.017, 0.154, 0.002, 0.595],
    features: [
      { name: "RETRANSMITTED_OUT_BYTES", phi: 0.4882 },
      { name: "DST_PORT_GROUP", phi: 0.2610 },
      { name: "NUM_PKTS_256_TO_512_BYTES", phi: -0.2576 },
      { name: "SRC_TO_DST_IAT_STDDEV", phi: -0.2544 },
      { name: "SRC_TO_DST_IAT_MAX", phi: -0.2417 },
      { name: "L7_PROTO", phi: 0.2057 },
    ],
    nInWindow: 1, nTotalNbrs: 1, topTemporalShap: 0.0045,
    temporalNbrs: [
      { eid: 2309767, dt: "t−29.1s", phi: 0.0045 },
    ],
    nodeShap: [0.0894, 0.0903, 0.1357, 0.3051, 0.0017, 0.0193, 0.0145, 0.0010, 0.0009],
    nodeIds: [28, 29, 30, 9, 11, 14, 10, 12, 4],
    phiNFrac: 0.146, sumPhiN: 0.658,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "Retransmitted-byte signature (φ=+0.49); only 1 in-window neighbour, the fewest in the set, with node novelty carrying 14.6% of |φ|.",
  },
  {
    attackClass: "DoS", eid: 1942014,
    predClass: "DoS", trueClass: "DoS", isCorrect: true,
    proba: 0.523,
    probaVec: [0.002, 0.000, 0.019, 0.523, 0.272, 0.017, 0.100, 0.052, 0.013, 0.001],
    features: [
      { name: "FTP_COMMAND_RET_CODE", phi: 0.2429 },
      { name: "RETRANSMITTED_OUT_BYTES", phi: 0.2152 },
      { name: "MAX_TTL", phi: 0.1413 },
      { name: "NUM_PKTS_256_TO_512_BYTES", phi: 0.1211 },
      { name: "NUM_PKTS_512_TO_1024_BYTES", phi: -0.0993 },
      { name: "RETRANSMITTED_IN_PKTS", phi: 0.0870 },
    ],
    nInWindow: 9, nTotalNbrs: 9, topTemporalShap: 0.0372,
    temporalNbrs: [
      { eid: 1942010, dt: "t−0.0s", phi: -0.0372 },
      { eid: 1942009, dt: "t−0.0s", phi: -0.0372 },
      { eid: 1942015, dt: "t−0.0s", phi: -0.0372 },
    ],
    nodeShap: [0.0468, 0.0342, 0.0959, 0.0390],
    nodeIds: [28, 27, 30, 14],
    phiNFrac: 0.103, sumPhiN: 0.216,
    srcNovelty: 0.0000, dstNovelty: 0.0000,
    narrative: "FTP-return-code signature (φ=+0.24); lowest-confidence case in the set (52.3%), with 9 in-window neighbours.",
  },
  {
    attackClass: "Analysis", eid: 2099604,
    predClass: "Analysis", trueClass: "Analysis", isCorrect: true,
    proba: 0.911,
    probaVec: [0.000, 0.911, 0.001, 0.022, 0.041, 0.002, 0.003, 0.019, 0.000, 0.000],
    features: [
      { name: "SRC_PORT_IS_EPHEMERAL", phi: 0.0329 },
      { name: "DST_TO_SRC_IAT_AVG", phi: 0.0307 },
      { name: "DST_TO_SRC_IAT_STDDEV", phi: 0.0263 },
      { name: "NUM_PKTS_256_TO_512_BYTES", phi: 0.0174 },
      { name: "ICMP_IPV4_TYPE", phi: 0.0158 },
      { name: "DST_TO_SRC_IAT_MAX", phi: 0.0143 },
    ],
    nInWindow: 4, nTotalNbrs: 4, topTemporalShap: 0.0264,
    temporalNbrs: [
      { eid: 2098511, dt: "t−40.4s", phi: 0.0264 },
      { eid: 2099188, dt: "t−15.6s", phi: 0.0257 },
      { eid: 2098637, dt: "t−35.9s", phi: 0.0254 },
    ],
    nodeShap: [0.0004, -0.0466, -0.0422, 0.0907, 0.0094, 0.0004, 0.0002, 0.0000, -0.0004, -0.0015],
    nodeIds: [27, 28, 29, 14, 4, 13, 17, 12, 9, 11],
    phiNFrac: 0.464, sumPhiN: 0.192,
    srcNovelty: 0.0000, dstNovelty: 0.0006,
    narrative: "Ephemeral-source-port signature (φ=+0.03) — the weakest top-feature magnitude in the set; node novelty carries 46.4% of |φ|, the highest share of any case, so the explanation is driven mainly by the flow's neighbourhood rather than its own features.",
  },
];

// Derived constants — computed once from CASE_STUDIES data
const MAX_NODES   = Math.max(...CASE_STUDIES.map(c => c.nodeIds.length));   // 10 (Analysis)
const MAX_T_ROWS  = 3;  // always render this many temporal rows

// Per-class attack signature text shown in the bottom bar
const CASE_SIGNATURES = {
  Exploits:  "RETRANSMITTED_IN_PKTS (φ=+1.09) — strongest top-feature magnitude in the set.",
  Recon:     "DST_PORT_GROUP (φ=+1.07) — port sweep across a 30-neighbour in-window burst, the densest in the set.",
  Shellcode: "PROTOCOL (φ=+0.87) — protocol-field signature identifies the delivery channel.",
  Generic:   "DNS_QUERY_TYPE (φ=+0.86) — highest-confidence case in the set (99.7%).",
  Fuzzers:   "NUM_PKTS_128_TO_256_BYTES (φ=+0.58) — packet-size-bucket signature.",
  Worms:     "RETRANSMITTED_OUT_BYTES (φ=+0.49) — fewest in-window neighbours (1) in the set; node novelty carries 14.6% of |φ|.",
  DoS:       "FTP_COMMAND_RET_CODE (φ=+0.24) — lowest-confidence case in the set (52.3%).",
  Analysis:  "SRC_PORT_IS_EPHEMERAL (φ=+0.03) — weakest top-feature magnitude; 46.4% of |φ| from node novelty, the highest share in the set.",
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
