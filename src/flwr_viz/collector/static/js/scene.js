// three.js view of a FedState, drawn like a comic: toon shading, ink outlines,
// clients as small characters, the server as a big round one, each model as a
// stack of books (one book per layer), messages as paper planes.
// Everything that repeats per client is instanced, so hundreds of clients stay cheap.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { mergeGeometries } from "three/addons/BufferGeometryUtils.js";
import { stateColor } from "./state.js";

const MAX_NODES = 1024;
const MAX_BOOKS = 6000;
const MAX_SERVER_BOOKS = 128;
const MAX_PLANES = 3000;
const MAX_GHOSTS = 2400;
const FULL_DETAIL_NODES = 40;      // above this, book stacks only for the focused / hovered client
const PLANE_SAMPLE_NODES = 150;
const LOW_DETAIL_NODES = 60;      // above this, characters use low-poly meshes and fewer outlines    // above this, paper planes are sampled
const FLOOR_H = 4.8;              // height of one storey of the tower
const RING_RADII = [7.5, 10.5];   // rings of clients on a storey once there is more than one storey
const SPACING = 2.1;              // space per client along a ring
const CLIENT_SCALE = 1.3;
const INK_HEX = 0x1f1d22;

// ------------------------------------------------------------------ materials
const TOON = (() => { const t = new THREE.DataTexture(new Uint8Array([120, 120, 120, 255, 195, 195, 195, 255, 255, 255, 255, 255]), 3, 1); t.minFilter = t.magFilter = THREE.NearestFilter; t.needsUpdate = true; return t; })();
const toon = (o = {}) => new THREE.MeshToonMaterial({ gradientMap: TOON, ...o });
const INK = new THREE.MeshBasicMaterial({ color: INK_HEX, side: THREE.BackSide });

// change colour ramp tuned for a light page: pale blue (tiny) -> mint -> gold -> coral -> red (huge)
const COLD = new THREE.Color(0xe4dccd);
const RAMP = [0xdbe6f3, 0x9cc5e8, 0x8fd3bd, 0xf2d27a, 0xf4a259, 0xe76f51, 0xc8323b].map(c => new THREE.Color(c));
export const RAMP_CSS = RAMP.map(c => "#" + c.getHexString());
export function changeColor(delta, norm, out = new THREE.Color()) {
  if (delta === undefined || delta === null || !(norm > 0)) return out.copy(COLD);
  const v = Math.min(1, Math.max(0, (Math.log10(Math.max(delta / norm, 1e-6)) + 4) / 4));
  const x = v * (RAMP.length - 1), i = Math.min(RAMP.length - 2, Math.floor(x));
  return out.copy(RAMP[i]).lerp(RAMP[i + 1], x - i);
}

const SKINS = [0xf6dcc6, 0xf0cdb0, 0xe6bd9a, 0xd7a47f].map(c => new THREE.Color(c));
const HAIRS = [0x232742, 0x5a3825, 0x1b1b1f, 0x9b4a2c, 0xd8b25a, 0x2f6f6a, 0x6b4c8a, 0x8a8f99].map(c => new THREE.Color(c));

// ------------------------------------------------------------------ helpers
const V3 = THREE.Vector3;
const tM = new THREE.Matrix4(), tM2 = new THREE.Matrix4(), tM3 = new THREE.Matrix4(), tQ = new THREE.Quaternion(),
  tE = new THREE.Euler(), tS = new V3(), tP = new V3(), tC = new THREE.Color(), UP = new V3(0, 1, 0);
const ease = x => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;

function canvasTex(w, h, draw) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

// Outline geometry: the same shape, a little bigger, drawn from its back faces in ink.
function hullGeo(geo, t) {
  const g = geo.clone(); g.computeBoundingBox();
  const bb = g.boundingBox, c = bb.getCenter(new V3()), s = bb.getSize(new V3());
  g.translate(-c.x, -c.y, -c.z);
  g.scale(1 + 2 * t / Math.max(s.x, 1e-3), 1 + 2 * t / Math.max(s.y, 1e-3), 1 + 2 * t / Math.max(s.z, 1e-3));
  g.translate(c.x, c.y, c.z);
  return g;
}

// An instanced part plus its ink hull, sharing one instance matrix buffer.
class Part {
  constructor(scene, geo, mat, cap, inkT = 0.022, colored = false) {
    this.mesh = new THREE.InstancedMesh(geo, mat, cap);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false; this.mesh.count = 0;
    if (colored) { this.mesh.setColorAt(0, COLD); this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage); }
    scene.add(this.mesh);
    if (inkT > 0) {
      this.hull = new THREE.InstancedMesh(hullGeo(geo, inkT), INK, cap);
      this.hull.instanceMatrix = this.mesh.instanceMatrix; // same buffer, no extra uploads
      this.hull.frustumCulled = false; this.hull.count = 0; this.hull.raycast = () => {};
      scene.add(this.hull);
    }
    this.n = 0;
  }
  begin() { this.n = 0; }
  push(m, color) {
    if (this.n >= this.mesh.instanceMatrix.count) return -1;
    this.mesh.setMatrixAt(this.n, m);
    if (color && this.mesh.instanceColor) this.mesh.setColorAt(this.n, color);
    return this.n++;
  }
  end() {
    this.mesh.count = this.n; if (this.hull) this.hull.count = this.n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor && this.n) this.mesh.instanceColor.needsUpdate = true;
  }
}

// ------------------------------------------------------------------ faces
// Face textures: u = 0.25 of a SphereGeometry is its +z side, where the face goes.
function faceTex(kind, { cheeks = true, mustache = false } = {}) {
  return canvasTex(1024, 512, (g, w, h) => {
    g.fillStyle = "#ffffff"; g.fillRect(0, 0, w, h);          // tinted by the instance's skin colour
    const cx = 256, ey = 262;
    if (cheeks) { g.fillStyle = "rgba(235,120,120,.38)"; [-82, 82].forEach(dx => { g.beginPath(); g.ellipse(cx + dx, ey + 58, 28, 12, 0, 0, 7); g.fill(); }); }
    g.strokeStyle = "#1f1d22"; g.fillStyle = "#1f1d22"; g.lineCap = "round"; g.lineJoin = "round";
    [-1, 1].forEach(sd => {
      const x = cx + sd * 52;
      if (kind === "happy") { g.lineWidth = 10; g.beginPath(); g.moveTo(x - 28, ey + 10); g.quadraticCurveTo(x, ey - 26, x + 28, ey + 10); g.stroke(); }
      else if (kind === "sleep") { g.lineWidth = 8; g.beginPath(); g.moveTo(x - 28, ey + 4); g.quadraticCurveTo(x, ey + 20, x + 28, ey + 4); g.stroke(); }
      else if (kind === "dizzy") { g.lineWidth = 9; g.beginPath(); g.moveTo(x - 20, ey - 22); g.lineTo(x + 20, ey + 22); g.moveTo(x + 20, ey - 22); g.lineTo(x - 20, ey + 22); g.stroke(); }
      else {
        g.fillStyle = "#fff"; g.beginPath(); g.ellipse(x, ey, 28, 38, 0, 0, 7); g.fill();
        const ir = g.createLinearGradient(0, ey - 32, 0, ey + 34); ir.addColorStop(0, "#1c2340"); ir.addColorStop(.6, "#2f6f8f"); ir.addColorStop(1, "#7fd0d8");
        g.fillStyle = ir; g.beginPath(); g.ellipse(x + sd * 2, ey + 4, 20, 30, 0, 0, 7); g.fill();
        g.fillStyle = "#11131f"; g.beginPath(); g.ellipse(x + sd * 2, ey + 4, 9, 14, 0, 0, 7); g.fill();
        g.fillStyle = "#fff"; g.beginPath(); g.ellipse(x - 7, ey - 11, 8, 10, -.3, 0, 7); g.fill();
        g.lineWidth = 8; g.beginPath(); g.moveTo(x - 34, ey - 28); g.quadraticCurveTo(x, ey - 50, x + 34, ey - 26); g.stroke();
      }
      g.lineWidth = 6;
      const browY = kind === "dizzy" ? ey - 62 : ey - 68;
      g.beginPath(); g.moveTo(x - 28, browY + (kind === "dizzy" ? sd * 6 : 0)); g.quadraticCurveTo(x, browY - 12, x + 26, browY - (kind === "dizzy" ? sd * 6 : 0)); g.stroke();
    });
    g.lineWidth = 3; g.beginPath(); g.moveTo(cx + 4, ey + 48); g.lineTo(cx, ey + 58); g.stroke();
    if (mustache) {
      g.fillStyle = "#3b2a22";
      [-1, 1].forEach(sd => { g.beginPath(); g.ellipse(cx + sd * 26, ey + 74, 30, 11, sd * .35, 0, 7); g.fill(); });
    }
    g.lineWidth = 5; g.beginPath();
    if (kind === "dizzy") { g.moveTo(cx - 24, ey + 96); for (let i = 0; i <= 6; i++) g.lineTo(cx - 24 + i * 8, ey + 96 + (i % 2 ? -7 : 7)); }
    else if (kind === "happy") { g.fillStyle = "#7a2b32"; g.moveTo(cx - 26, ey + 86); g.quadraticCurveTo(cx, ey + 122, cx + 26, ey + 86); g.closePath(); g.fill(); }
    else if (kind === "sleep") { g.arc(cx, ey + 92, 7, 0, 7); }
    else { g.moveTo(cx - 16, ey + 88); g.quadraticCurveTo(cx, ey + 100, cx + 18, ey + 86); }
    g.stroke();
  });
}

// ------------------------------------------------------------------ client geometry (feet at y=0, facing +z)
function clientGeometries(lo = false) {
  const q = (hi, low) => lo ? low : hi;   // segment counts for the two levels of detail
  const leg = new THREE.CapsuleGeometry(.085, .26, q(4, 2), q(8, 5)); leg.translate(0, -.215, 0);
  const shoe = new THREE.BoxGeometry(.15, .09, .25); shoe.translate(0, -.44, .04);
  const body = new THREE.CapsuleGeometry(.2, .24, q(4, 2), q(12, 7)); body.scale(1, 1, .85); body.translate(0, .29, 0);
  const seat = new THREE.CylinderGeometry(.215, .195, .2, q(14, 7)); seat.scale(1, 1, .86); seat.translate(0, .06, 0);
  const arm = new THREE.CapsuleGeometry(.06, .24, q(4, 2), q(8, 5)); arm.translate(0, -.18, 0);
  const hand = new THREE.SphereGeometry(.065, q(10, 5), q(8, 4)); hand.translate(0, -.36, 0);
  const head = new THREE.SphereGeometry(.34, q(28, 14), q(18, 9));
  // spiky hair: a cap plus cones swept back, merged into one geometry
  const parts = [];
  const cap = new THREE.SphereGeometry(.365, q(20, 10), q(12, 5), 0, Math.PI * 2, 0, Math.PI * .52); cap.rotateX(-.22); cap.translate(0, .04, -.03); parts.push(cap);
  const back = new THREE.SphereGeometry(.35, q(16, 8), q(12, 6)); back.scale(1, .92, .8); back.translate(0, -.02, -.1); parts.push(back);
  const spike = (x, y, z, r, len, rx, rz) => { const c = new THREE.ConeGeometry(r, len, q(7, 4)); c.rotateX(rx); c.rotateZ(rz); c.translate(x, y, z); parts.push(c); };
  [[-.2, .27, -.1, .12, .36, -.5, .7], [0, .34, -.13, .13, .4, -.7, 0], [.2, .27, -.1, .12, .36, -.5, -.7], [-.28, .08, -.2, .1, .32, -1.2, 1.1], [.28, .08, -.2, .1, .32, -1.2, -1.1], [0, .1, -.33, .12, .36, -1.6, 0],
   [-.21, .13, .22, .08, .28, 2.5, .35], [-.07, .18, .28, .085, .3, 2.65, .12], [.08, .18, .28, .085, .28, 2.65, -.15], [.22, .13, .22, .08, .26, 2.5, -.4]].forEach(a => spike(...a));
  const hair = mergeGeometries(parts.map(p => p.index ? p.toNonIndexed() : p));
  return { leg, shoe, body, seat, arm, hand, head, hair };
}

function planeGeometry() {
  const p = [0, 0, .2, -.13, .01, -.12, 0, 0, -.08, 0, 0, .2, 0, 0, -.08, .13, .01, -.12, 0, 0, .2, 0, -.06, -.1, 0, 0, -.08];
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); g.computeVertexNormals(); return g;
}

// ------------------------------------------------------------------ poses
// Returns joint angles for a character in a given state at time t.
function pose(st, t, seed) {
  const p = { bob: 0, squash: 1, tilt: 0, legL: 0, legR: 0, armL: [0, -.12], armR: [0, .12], headX: 0, headZ: 0, headY: 0 };
  const ph = t + seed * 6.28;
  switch (st) {
    case "idle": p.squash = 1 + .015 * Math.sin(ph * 2); p.headZ = .06 * Math.sin(ph * .7); break;
    case "selected": p.armR = [-.3, 2.5 + .35 * Math.sin(ph * 9)]; p.headZ = -.08; p.bob = .02 * Math.abs(Math.sin(ph * 4)); break;
    case "training": {
      const s = Math.sin(ph * 11);
      p.legL = .55 * s; p.legR = -.55 * s; p.armL = [-.7 * s, -.2]; p.armR = [.7 * s, .2];
      p.bob = .06 * Math.abs(Math.cos(ph * 11)); p.tilt = .12; break;
    }
    case "evaluating": p.armR = [-1.9, .5]; p.armL = [-.2, -.35]; p.headZ = .18 * Math.sin(ph * 1.6); p.headX = .08; break;
    case "working": { const s = Math.sin(ph * 14); p.armL = [-1.25 + .18 * s, -.15]; p.armR = [-1.25 - .18 * s, .15]; p.headX = .22; p.tilt = .06; break; }   // busy at a keyboard
    case "sending": { const s = Math.sin(ph * 7); p.armL = [-2.2 + .4 * s, -.25]; p.armR = [-2.2 + .4 * s, .25]; p.bob = .05 * Math.abs(s); break; }
    case "done": { p.bob = .14 * Math.abs(Math.sin(ph * 6)); p.armL = [0, -2.5]; p.armR = [0, 2.5]; break; }
    case "failed": p.tilt = .16; p.headX = -.08; p.headZ = .28 * Math.sin(ph * 2.5); p.armL = [.1, -.35]; p.armR = [.1, .35]; p.squash = .95; break;   // dazed, swaying
    case "offline": p.squash = .88; p.tilt = .25; p.headX = .45; break;
    default: break;
  }
  return p;
}

function faceFor(st) { return st === "failed" ? 3 : st === "done" || st === "sending" ? 1 : st === "offline" ? 2 : 0; }

function hash(s) { let h = 2166136261; s = String(s); for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

// ------------------------------------------------------------------ the scene
export class FedScene {
  constructor(container, { onPick, onHover } = {}) {
    this.container = container; this.onPick = onPick; this.onHover = onHover;
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.setClearColor(0xf4efe4);
    container.appendChild(r.domElement);
    const sc = this.scene = new THREE.Scene();
    sc.fog = new THREE.Fog(0xf4efe4, 40, 140);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 600);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true; this.controls.maxPolarAngle = Math.PI * 0.47; this.controls.minDistance = 3;
    this.camGoal = null;

    sc.add(new THREE.HemisphereLight(0xfff8ee, 0xd8c8b0, 1.6));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6); dl.position.set(6, 14, 9); sc.add(dl);

    // floor: warm paper ground and a tiled plaza
    const ground = this.ground = new THREE.Mesh(new THREE.CircleGeometry(300, 64), new THREE.MeshBasicMaterial({ color: 0xefe2cf }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.03; sc.add(ground);
    this.floorGroup = new THREE.Group(); sc.add(this.floorGroup); this.floors = 1;
    this.tileTex = canvasTex(256, 256, (g, w, h) => {
      g.fillStyle = "#f6f1e6"; g.fillRect(0, 0, w, h); g.fillStyle = "#cfe8dd"; g.fillRect(0, 0, w / 2, h / 2); g.fillRect(w / 2, h / 2, w / 2, h / 2);
      g.fillStyle = "rgba(31,29,34,.22)"; g.fillRect(0, 0, w, 2); g.fillRect(0, 0, 2, h); g.fillRect(0, h / 2, w, 2); g.fillRect(w / 2, 0, 2, h);
    });
    this.tileTex.wrapS = this.tileTex.wrapT = THREE.RepeatWrapping;
    this.serverRug = new THREE.Mesh(new THREE.CircleGeometry(2.3, 64), new THREE.MeshBasicMaterial({ color: 0xf3d3c4 }));
    this.serverRug.rotation.x = -Math.PI / 2; this.serverRug.position.y = .006; sc.add(this.serverRug);
    const rugRim = new THREE.Mesh(new THREE.RingGeometry(2.3, 2.36, 96), new THREE.MeshBasicMaterial({ color: INK_HEX }));
    rugRim.rotation.x = -Math.PI / 2; rugRim.position.y = .008; sc.add(rugRim);
    this.aggRing = new THREE.Mesh(new THREE.RingGeometry(2.45, 2.62, 96), new THREE.MeshBasicMaterial({ color: 0xf2b84b, transparent: true, opacity: 0 }));
    this.aggRing.rotation.x = -Math.PI / 2; this.aggRing.position.y = .01; sc.add(this.aggRing);

    // --- clients
    const skinMat = toon({ color: 0xffffff }), shirtMat = toon({ color: 0xffffff }), trouserMat = toon({ color: 0x2c3554 }),
      shoeMat = toon({ color: 0xd4483f }), hairMat = toon({ color: 0xffffff });
    this.faces = ["normal", "happy", "sleep", "dizzy"].map(k => faceTex(k));
    const charSet = (G, lo) => {
      const limbInk = lo ? 0 : 1;   // in a crowd, only body, head and hair keep outlines
      const parts = {
        legL: new Part(sc, G.leg, trouserMat, MAX_NODES, .022 * limbInk), legR: new Part(sc, G.leg, trouserMat, MAX_NODES, .022 * limbInk),
        shoeL: new Part(sc, G.shoe, shoeMat, MAX_NODES, .022 * limbInk), shoeR: new Part(sc, G.shoe, shoeMat, MAX_NODES, .022 * limbInk),
        seat: new Part(sc, G.seat, trouserMat, MAX_NODES, .022 * limbInk),
        body: new Part(sc, G.body, shirtMat, MAX_NODES, .022, true),
        armL: new Part(sc, G.arm, shirtMat, MAX_NODES, .02 * limbInk, true), armR: new Part(sc, G.arm, shirtMat, MAX_NODES, .02 * limbInk, true),
        handL: new Part(sc, G.hand, skinMat, MAX_NODES, .018 * limbInk, true), handR: new Part(sc, G.hand, skinMat, MAX_NODES, .018 * limbInk, true),
        headInk: new Part(sc, G.head, INK, MAX_NODES, .026),   // only its hull is used
        hair: new Part(sc, G.hair, hairMat, MAX_NODES, lo ? .022 : .018, true),
      };
      parts.headInk.mesh.visible = false;
      // one head mesh per expression, each with its own face texture
      const heads = this.faces.map(tex => new Part(sc, G.head, toon({ map: tex }), MAX_NODES, 0, true));
      return { parts, heads };
    };
    this.sets = { hi: charSet(clientGeometries(false), false), lo: charSet(clientGeometries(true), true) };
    this.cs = this.sets.hi;
    this.shadows = new Part(sc, new THREE.CircleGeometry(.46, 28).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: .12, depthWrite: false }), MAX_NODES, 0);
    this.halos = new Part(sc, new THREE.RingGeometry(.5, .64, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ transparent: true, opacity: .9, depthWrite: false }), MAX_NODES, 0, true);
    this.barBg = new Part(sc, new THREE.BoxGeometry(1, 1, 1), toon({ color: 0xfffdf7 }), MAX_NODES, .012);
    this.barFill = new Part(sc, new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x3fa58d }), MAX_NODES, 0);

    // books (layers), server pile, flying ghost books, paper planes, failure rings
    const book = new THREE.BoxGeometry(1, 1, 1);
    this.books = new Part(sc, book, toon({ color: 0xffffff }), MAX_BOOKS, .018, true);
    this.pages = new Part(sc, book, toon({ color: 0xfffdf7 }), MAX_BOOKS, 0);
    this.serverBooks = new Part(sc, book, toon({ color: 0xffffff }), MAX_SERVER_BOOKS, .02, true);
    this.serverPages = new Part(sc, book, toon({ color: 0xfffdf7 }), MAX_SERVER_BOOKS, 0);
    this.ghosts = new Part(sc, book, toon({ color: 0xffffff, transparent: true, opacity: .92 }), MAX_GHOSTS, .015, true);
    // data chests (learn mode): each client's data, locked where it lives
    const chestBox = new THREE.BoxGeometry(.5, .3, .36); chestBox.translate(0, .15, 0);
    const chestLid = new THREE.BoxGeometry(.54, .1, .4); chestLid.translate(0, .35, 0);
    const chestLock = new THREE.BoxGeometry(.1, .12, .04); chestLock.translate(0, .26, .2);
    this.chestMats = [toon({ color: 0x9b6a3c }), toon({ color: 0x7a4f2a }), toon({ color: 0xe2b34a })];
    this.chests = [new Part(sc, chestBox, this.chestMats[0], MAX_NODES, .02), new Part(sc, chestLid, this.chestMats[1], MAX_NODES, .02), new Part(sc, chestLock, this.chestMats[2], MAX_NODES, 0)];
    this.showChests = false; this.lensOn = false; this.highlightIds = new Set(); this.bookLayer = []; this.lastPickLayer = null;
    this.planes = new Part(sc, planeGeometry(), toon({ color: 0xffffff, side: THREE.DoubleSide }), MAX_PLANES, 0, true);
    this.bursts = new Part(sc, new THREE.RingGeometry(.25, .38, 24).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xe5534b, transparent: true, opacity: .85, depthWrite: false }), 512, 0);

    // links from the server to busy clients
    this.linkPos = new Float32Array(MAX_NODES * 6);
    this.linkGeo = new THREE.BufferGeometry(); this.linkGeo.setAttribute("position", new THREE.BufferAttribute(this.linkPos, 3));
    this.links = new THREE.LineSegments(this.linkGeo, new THREE.LineDashedMaterial({ color: INK_HEX, transparent: true, opacity: .28, dashSize: .25, gapSize: .18 }));
    sc.add(this.links);

    // the projection beam that rises above the selected character
    this.beam = new THREE.Group(); this.beam.visible = false; sc.add(this.beam);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(.8, .12, 1, 40, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x7cc4f2, transparent: true, opacity: .2, side: THREE.DoubleSide, depthWrite: false }));
    cone.position.y = .5; this.beam.add(cone); this.beamCone = cone;
    const disc = new THREE.Mesh(new THREE.TorusGeometry(.16, .035, 8, 24), new THREE.MeshBasicMaterial({ color: 0x3d9bd8 }));
    disc.rotation.x = Math.PI / 2; this.beam.add(disc);
    this.beamTop = new THREE.Mesh(new THREE.TorusGeometry(.8, .025, 6, 64), new THREE.MeshBasicMaterial({ color: 0x3d9bd8, transparent: true, opacity: .55 }));
    this.beamTop.rotation.x = Math.PI / 2; this.beam.add(this.beamTop);
    this.beamH = 0.6;

    this._buildPillar();

    this.positions = new Map(); this.layoutN = -1; this.outerR = 8;
    this.focus = null; this.hover = null; this.podIds = []; this.headIndex = [];
    this.bookIndex = []; this.pickIndex = [];
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2(); this.mouseMoved = false;
    const el = r.domElement;
    el.addEventListener("pointermove", e => { this._setMouse(e); this.mouseMoved = true; });
    let down = null;
    el.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; this.camGoal = null; });
    el.addEventListener("wheel", () => { this.camGoal = null; }, { passive: true });
    el.addEventListener("pointerup", e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      this._setMouse(e); this.onPick && this.onPick(this._pick());
    });
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  // ---------------------------------------------------------------- server: a floating diamond with a ring around it
  _buildPillar() {
    const g = this.server = new THREE.Group(); this.scene.add(g);
    const P = this.sp = {};
    P.coreY = 2.7;
    P.diamondMat = toon({ color: 0x5fb4f0 });
    const geo = new THREE.OctahedronGeometry(.85, 0); geo.scale(1, 1.45, 1);
    P.diamond = new THREE.Mesh(geo, P.diamondMat); g.add(P.diamond);
    // a lighter facet on top so it reads as a cut gem
    const capGeo = new THREE.OctahedronGeometry(.55, 0); capGeo.scale(1, .5, 1);
    P.facet = new THREE.Mesh(capGeo, toon({ color: 0xbfe6ff })); P.facet.position.y = .55; P.diamond.add(P.facet); P.facet.userData.noInk = true;
    const glowTex = canvasTex(128, 128, (c, w, h) => { const gr = c.createRadialGradient(64, 64, 4, 64, 64, 64); gr.addColorStop(0, "rgba(124,196,242,.85)"); gr.addColorStop(1, "rgba(124,196,242,0)"); c.fillStyle = gr; c.fillRect(0, 0, w, h); });
    P.glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, transparent: true, depthWrite: false }));
    g.add(P.glow);
    P.ringMat = toon({ color: 0xe2b34a });
    P.ring = new THREE.Mesh(new THREE.TorusGeometry(1.75, .08, 10, 96), P.ringMat); P.ring.rotation.x = Math.PI / 2; g.add(P.ring);
    this._inkAll(g);
    this.serverPickables = []; g.traverse(o => { if (o.isMesh && !o.userData.ink) this.serverPickables.push(o); });
    this._setPillar(0);
  }

  _inkAll(root) {
    const list = []; root.traverse(o => { if (o.isMesh && !o.userData.ink && !o.userData.inked && !o.userData.noInk) list.push(o); });
    for (const o of list) {
      o.geometry.computeBoundingBox(); const s = o.geometry.boundingBox.getSize(new V3());
      if (Math.min(s.x, s.y, s.z) < .05) continue;
      const hull = new THREE.Mesh(hullGeo(o.geometry, .025), INK); hull.userData.ink = true; hull.raycast = () => {}; o.add(hull); o.userData.inked = true;
    }
  }

  // The diamond floats above the centre of the top storey.
  _setPillar() {
    const P = this.sp;
    P.diamond.position.y = P.coreY; P.ring.position.y = P.coreY; P.glow.position.y = P.coreY;
    this.serverTop = P.coreY + 1.35; this.serverEyeY = P.coreY;
  }

  _setMouse(e) {
    const b = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
  }

  _pick() {
    this.ray.setFromCamera(this.mouse, this.camera);
    const cs = this.cs;
    const targets = [cs.parts.body.mesh, cs.parts.hair.mesh, ...cs.heads.map(h => h.mesh), this.books.mesh, ...this.serverPickables, this.serverBooks.mesh];
    // InstancedMesh caches one bounding sphere over all instances; ours move every frame
    for (const m of targets) if (m.isInstancedMesh) { m.boundingSphere = null; m.boundingBox = null; }
    const hits = this.ray.intersectObjects(targets, false);
    if (!hits.length) return null;
    const h = hits[0];
    if (this.serverPickables.includes(h.object) || h.object === this.serverBooks.mesh) return "server";
    this.lastPickLayer = null;
    if (h.object === this.books.mesh) { this.lastPickLayer = this.bookLayer[h.instanceId] ?? null; return this.bookIndex[h.instanceId] ?? null; }
    const hi = cs.heads.findIndex(p => p.mesh === h.object);
    if (hi >= 0) return this.headIndex[hi][h.instanceId] ?? null;
    return this.podIds[h.instanceId] ?? null;  // body and hair are pushed in node order
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h); this.camera.updateProjectionMatrix();
  }

  overview() {
    const towerH = (this.floors - 1) * FLOOR_H;
    const d = Math.max(this.outerR * 2.3 + 10, towerH * 1.75 + 30);
    const ty = 1.4 - towerH / 2;
    return { target: new V3(0, ty, 0), pos: new V3(0, ty + d * (this.floors > 1 ? .42 : .6), d * .9) };
  }

  // Glide the camera to a node (or back to the overview with null).
  focusCamera(id) {
    if (id === null || id === undefined) { this.camGoal = this.overview(); return; }
    const p = id === "server" ? new V3(0, 0, 0) : this.pos(id);
    if (!p) return;
    // from outside the ring and above (the selected character turns round to face the camera),
    // aimed above the head so the character sits low on screen and the projection has room
    const srv = id === "server";
    const target = new V3(p.x, srv ? this.serverTop + 1.6 : p.y + 4.4, p.z);
    const out = srv ? new V3(0, 0, 1) : new V3(p.x, 0, p.z).normalize();
    const up = srv ? 5 : this.floors > 1 ? 3.6 : 6.5;      // stay level with the storey in a tower
    this.camGoal = { target, pos: target.clone().addScaledVector(out, srv ? 10 : this.floors > 1 ? 10 : 8.5).add(new V3(0, up, 0)) };
  }

  layout(order) {
    const n = order.length;
    if (n === this.layoutN) return;
    this.layoutN = n; this.positions.clear();
    const caps = RING_RADII.map(r => Math.floor(2 * Math.PI * r / SPACING));
    const perStorey = caps.reduce((a, b) => a + b, 0);
    let floors = 1, rings;
    const single = Math.max(5.5, Math.min(8.5, 4 + n * 0.28));
    if (n <= Math.floor(2 * Math.PI * single / SPACING)) {
      rings = [[single, n]];                                     // everyone fits on one ring
    } else {
      floors = Math.ceil(n / perStorey);                         // otherwise: storeys stacked downwards
    }
    this.floors = floors;
    let i = 0;
    for (let f = 0; f < floors && i < n; f++) {
      const here = floors === 1 && rings ? n : Math.min(n - i, Math.ceil(n / floors));
      const fr = rings || [[RING_RADII[0], Math.min(here, caps[0])], [RING_RADII[1], Math.max(0, here - caps[0])]];
      fr.forEach(([r, k], ri) => {
        for (let j = 0; j < k && i < n; j++, i++) {
          const a = (j / k) * Math.PI * 2 + ri * 0.37 + f * 0.21 + Math.PI / 2;
          this.positions.set(String(order[i]), new V3(Math.cos(a) * r, -f * FLOOR_H, Math.sin(a) * r));
        }
      });
    }
    this.outerR = floors > 1 ? RING_RADII[RING_RADII.length - 1] : single;
    this._buildFloors();
    this._setPillar();
    this.ground.position.y = -(floors - 1) * FLOOR_H - .03;
    const ov = this.overview();
    if (!this.focus) { this.camera.position.copy(ov.pos); this.controls.target.copy(ov.target); }
    const d = ov.pos.distanceTo(ov.target);
    this.scene.fog.near = d * 1.3; this.scene.fog.far = d * 3.5;
  }

  // One tiled storey plate per floor. Lower storeys show through the ones above.
  _buildFloors() {
    for (const c of [...this.floorGroup.children]) { this.floorGroup.remove(c); c.geometry?.dispose(); }
    const R = this.outerR + 2.4, multi = this.floors > 1;
    this.tileTex.repeat.set(R / 1.6, R / 1.6);
    for (let f = 0; f < this.floors; f++) {
      const y = -f * FLOOR_H;
      const plate = new THREE.Mesh(new THREE.CircleGeometry(R, 96),
        new THREE.MeshBasicMaterial({ map: this.tileTex, transparent: multi, opacity: multi ? (f === 0 ? .82 : .62) : 1, depthWrite: !multi, side: THREE.DoubleSide }));
      plate.rotation.x = -Math.PI / 2; plate.position.y = y; plate.renderOrder = -1; this.floorGroup.add(plate);
      const rim = new THREE.Mesh(new THREE.TorusGeometry(R, .06, 6, 128), new THREE.MeshBasicMaterial({ color: INK_HEX }));
      rim.rotation.x = Math.PI / 2; rim.position.y = y; this.floorGroup.add(rim);
      if (multi) {
        const tag = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(256, 96, (g, w, h) => {
          g.fillStyle = "#fffdf7"; g.strokeStyle = "#1f1d22"; g.lineWidth = 6; g.beginPath(); g.roundRect(6, 6, w - 12, h - 12, 12); g.fill(); g.stroke();
          g.fillStyle = "#1f1d22"; g.font = '46px "Patrick Hand", "Comic Sans MS", cursive'; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(`FLOOR ${f + 1}`, w / 2, h / 2 + 2);
        }), depthWrite: false }));
        tag.scale.set(2.4, .9, 1); tag.position.set(-R - .6, y + .9, 0); this.floorGroup.add(tag);
      }
    }
    // the server's rug and the aggregation ring sit on the top storey
    this.serverRug.visible = true;
  }

  pos(id) { return id === "server" ? new V3(0, 0, 0) : this.positions.get(String(id)); }

  // world point the projection card hangs from (top of the beam)
  anchor(id) {
    if (id === "server") return new V3(0, this.serverTop + this.beamH, 0);
    const p = this.pos(id); return p ? new V3(p.x, p.y + 2.95 + this.beamH, p.z) : null;
  }

  _stack(layers, scale, maxH) {
    const max = Math.max(1, ...layers.map(l => l.params || 0));
    const h = Math.min(.12, maxH / Math.max(1, layers.length)) * scale;
    let y = 0;
    return layers.map((l, i) => {
      const w = (.26 + .5 * Math.sqrt((l.params || 0) / max)) * scale;
      const item = { w, d: w * .72 + .06 * scale, h: h * .86, y: y + h * .43, l, rot: ((hash(l.name + i) % 100) / 100 - .5) * .35 };
      y += h; return item;
    });
  }

  _pushBook(part, pages, x, y, z, b, yaw, s = 1, color) {
    tQ.setFromAxisAngle(UP, yaw + b.rot);
    tM.compose(tP.set(x, y, z), tQ, tS.set(b.w * s, b.h * s, b.d * s));
    part.push(tM, color);
    // the page block: slightly shorter, peeking out on one side
    tM2.compose(tP.set(0, 0, .06), new THREE.Quaternion(), tS.set(.9, .7, .9));
    tM3.multiplyMatrices(tM, tM2);
    pages.push(tM3);
  }

  update(state, tl, vt, anim, speed) {
    const order = state.order;
    this.layout(order);
    const N = order.length;
    this.podIds = order;
    const D = 0.9 * Math.max(1, speed);
    const cs = this.cs = N > LOW_DETAIL_NODES ? this.sets.lo : this.sets.hi;
    const other = cs === this.sets.hi ? this.sets.lo : this.sets.hi;
    for (const k in other.parts) { other.parts[k].begin(); other.parts[k].end(); }
    other.heads.forEach(h => { h.begin(); h.end(); });
    const P = cs.parts;
    for (const k in P) P[k].begin();
    cs.heads.forEach(h => h.begin()); this.headIndex = cs.heads.map(() => []);
    for (const c of this.chests) c.begin();
    this.shadows.begin(); this.halos.begin(); this.barBg.begin(); this.barFill.begin();

    let li = 0;
    for (let i = 0; i < N; i++) {
      const id = String(order[i]), n = state.nodes[order[i]], p = this.pos(id);
      const st = n.state, h = hash(id), seed = (h % 1000) / 1000;
      const q = pose(st, anim, seed);
      const yaw = this.focus === id || this.tourFocus === id
        ? Math.atan2(this.camera.position.x - p.x, this.camera.position.z - p.z)   // turns to face you
        : Math.atan2(-p.x, -p.z);                                                   // otherwise faces the server
      const big = CLIENT_SCALE * ((this.hover === id || this.focus === id) ? 1.1 : 1);
      // character frame
      tQ.setFromAxisAngle(UP, yaw);
      const C = new THREE.Matrix4().compose(tP.set(p.x, p.y + q.bob, p.z), tQ, tS.setScalar(big));
      // legs (+ shoes) hang from the hips
      for (const [part, shoe, sx, ang] of [[P.legL, P.shoeL, -.1, q.legL], [P.legR, P.shoeR, .1, q.legR]]) {
        tM.multiplyMatrices(C, tM2.compose(tP.set(sx, .5, 0), tQ.setFromEuler(tE.set(ang, 0, 0)), tS.set(1, 1, 1)));
        part.push(tM); shoe.push(tM);
      }
      // torso frame: tilts forward and squashes
      const T = new THREE.Matrix4().multiplyMatrices(C, tM2.compose(tP.set(0, .5, 0), tQ.setFromEuler(tE.set(q.tilt, 0, 0)), tS.set(1, q.squash, 1)));
      tC.setHex(stateColor(st));
      const shirt = tC.clone();
      P.body.push(T, shirt); P.seat.push(T);
      const skin = SKINS[h % SKINS.length], hair = HAIRS[(h >>> 3) % HAIRS.length];
      for (const [arm, hand, sx, a] of [[P.armL, P.handL, -.27, q.armL], [P.armR, P.handR, .27, q.armR]]) {
        tM.multiplyMatrices(T, tM2.compose(tP.set(sx, .48, 0), tQ.setFromEuler(tE.set(a[0], 0, a[1])), tS.set(1, 1, 1)));
        arm.push(tM, shirt); hand.push(tM, skin);
      }
      const Hm = new THREE.Matrix4().multiplyMatrices(T, tM2.compose(tP.set(0, .92, 0), tQ.setFromEuler(tE.set(q.headX, q.headY, q.headZ)), tS.set(1, 1, 1)));
      const fi = faceFor(st);
      cs.heads[fi].push(Hm, skin); this.headIndex[fi].push(id);
      P.headInk.push(Hm); P.hair.push(Hm, hair);
      // ground: soft shadow + state halo
      tM.compose(tP.set(p.x, p.y + .012, p.z), tQ.identity(), tS.setScalar(CLIENT_SCALE));
      this.shadows.push(tM);
      if (this.showChests) {
        const cy = Math.atan2(-p.x, -p.z), side = new V3(-Math.cos(cy), 0, Math.sin(cy));   // the character's other side from its books
        tQ.setFromAxisAngle(UP, cy);
        tM.compose(tP.set(p.x + side.x * .85, p.y, p.z + side.z * .85), tQ, tS.setScalar(CLIENT_SCALE * .9));
        for (const c of this.chests) c.push(tM);
      }
      const lit = this.highlightIds.has(id);
      const hs = lit ? big * (1.2 + .12 * Math.sin(anim * 6)) : st === "idle" || st === "offline" ? 0.0001 : big;
      tM.compose(tP.set(p.x, p.y + .016, p.z), tQ.identity(), tS.setScalar(hs));
      this.halos.push(tM, lit ? new THREE.Color(0xf2b84b) : shirt);
      // progress bar over the head (only with report_progress data)
      if (st === "training" && n.progress.length) {
        const last = n.progress[n.progress.length - 1];
        const frac = last.total ? Math.min(1, (last.batch + 1) / last.total) : .5;
        tQ.setFromAxisAngle(UP, yaw);
        tM.compose(tP.set(p.x, p.y + 3.0 + q.bob, p.z), tQ, tS.set(.72, .09, .09)); this.barBg.push(tM);
        const off = new V3(-.36 + .36 * frac, 0, .05).applyQuaternion(tQ);
        tM.compose(tP.set(p.x + off.x, p.y + 3.0 + q.bob, p.z + off.z), tQ, tS.set(.72 * frac, .07, .03)); this.barFill.push(tM);
      }
      if (st === "selected" || st === "training" || st === "evaluating" || st === "working" || st === "sending") {
        this.linkPos.set([0, this.sp.coreY, 0, p.x, p.y + .05, p.z], li * 6); li++;
      }
    }
    for (const k in P) P[k].end();
    cs.heads.forEach(h => h.end());
    for (const c of this.chests) c.end();
    this.shadows.end(); this.halos.end(); this.barBg.end(); this.barFill.end();
    this.linkGeo.setDrawRange(0, li * 2); this.linkGeo.attributes.position.needsUpdate = true;
    if (li) this.links.computeLineDistances();

    // ---- the diamond: spins and glows while merging, flashes when a new model is ready
    const sv = state.server, S = this.sp;
    const flash = this._flash(tl, vt, D);
    const agg = sv.status === "aggregating", waiting = sv.status === "waiting";
    const spin = agg ? 6 : waiting ? 1.4 : .6;
    S.diamond.rotation.y = (S.diamond.rotation.y + .012 * spin) % (Math.PI * 2);
    const bob = .12 * Math.sin(anim * 1.6);
    S.diamond.position.y = S.coreY + bob; S.glow.position.y = S.coreY + bob;
    S.ring.rotation.z = (S.ring.rotation.z + .008 * spin) % (Math.PI * 2);
    S.ring.scale.setScalar(agg ? 1 + .06 * Math.sin(anim * 10) : 1 + flash * .15);
    S.diamond.scale.setScalar(1 + flash * .15);
    S.glow.material.opacity = agg ? .75 + .25 * Math.sin(anim * 9) : flash > 0 ? 1 : .55;
    S.glow.scale.setScalar(3.6 + (agg ? .6 * Math.sin(anim * 9) : 0) + flash * 2);
    S.diamondMat.color.setHex(state.run.error ? 0xe5534b : agg ? 0x7cc8ff : 0x5fb4f0);
    S.ringMat.color.setHex(agg ? 0xf2c14e : 0xe2b34a);
    this.aggRing.material.opacity = agg ? .55 + .35 * Math.sin(anim * 10) : flash * .8;

    // ---- server's book pile (the global model) to his right
    this.serverBooks.begin(); this.serverPages.begin();
    const sStack = this._stack(sv.layers, 1.6, 2.6);
    this.serverPile = new V3(2.9, 0, 1.2);
    for (const b of sStack) {
      changeColor(b.l.delta_norm, b.l.norm, tC);
      if (flash > 0) tC.lerp(new THREE.Color(0xffffff), flash * .55);
      this._pushBook(this.serverBooks, this.serverPages, this.serverPile.x, b.y, this.serverPile.z, b, -.5, 1, tC);
    }
    this.serverBooks.end(); this.serverPages.end();

    // ---- client book stacks (LOD)
    this.books.begin(); this.pages.begin(); this.bookIndex.length = 0;
    const showAll = N <= FULL_DETAIL_NODES;
    for (let i = 0; i < N; i++) {
      const id = String(order[i]);
      if (!showAll && id !== this.focus && id !== this.hover && id !== this.tourFocus) continue;
      const n = state.nodes[order[i]];
      const layers = n.layers.length ? n.layers : n.recvLayers;
      if (!layers.length) continue;
      const p = this.pos(id), yaw = Math.atan2(-p.x, -p.z);
      const right = new V3(Math.cos(yaw), 0, -Math.sin(yaw));
      const bx = p.x + right.x * .8, bz = p.z + right.z * .8;
      const fresh = n.layers.length > 0;
      for (const b of this._stack(layers, .85, 1.0).map((b, k) => ({ ...b, k }))) {
        if (fresh) changeColor(b.l.delta_norm, b.l.norm, tC); else tC.copy(COLD);
        if (n.state === "failed") tC.lerp(new THREE.Color(0xe5534b), .45);
        if (this.books.n >= MAX_BOOKS) break;
        this.bookIndex[this.books.n] = id; this.bookLayer[this.books.n] = b.k;
        this._pushBook(this.books, this.pages, bx, p.y + b.y, bz, b, yaw, 1, tC);
      }
    }
    this.books.end(); this.pages.end();

    this._planes(state, tl, vt, D, N);
    this._ghosts(state, tl, vt, D, sStack);
    this._beam(anim);
  }

  _beam(anim) {
    const id = this.focus;
    const a = id ? this.anchor(id) : null;
    this.beam.visible = !!a;
    if (!a) return;
    const base = id === "server" ? this.serverTop + .05 : a.y - this.beamH;
    this.beam.position.set(a.x, base, a.z);
    this.beamCone.scale.set(1, this.beamH, 1);
    this.beamTop.position.y = this.beamH;
    this.beamCone.material.opacity = .16 + .05 * Math.sin(anim * 3);
    const s = id === "server" ? 1.5 : 1; this.beam.scale.set(s, 1, s);
  }

  _flash(tl, vt, D) {
    const ix = tl.index; if (!ix) return 0;
    let best = 0;
    for (let k = ix.aggEnds.length - 1; k >= 0; k--) {
      const t = ix.aggEnds[k];
      if (t > vt) continue;
      if (vt - t > D) break;
      best = Math.max(best, 1 - (vt - t) / D);
    }
    return best;
  }

  _arc(a, b, f, out) {
    out.lerpVectors(a, b, f);
    out.y += Math.sin(Math.PI * f) * (1.4 + 0.1 * a.distanceTo(b));
    return out;
  }

  _planes(state, tl, vt, D, N) {
    const ix = tl.index;
    this.planes.begin(); this.bursts.begin();
    if (ix) {
      const stride = N > PLANE_SAMPLE_NODES ? Math.ceil(N / PLANE_SAMPLE_NODES) : 1;
      const srv = new V3(), A = new V3(), B = new V3(), B2 = new V3();
      const scan = (list, startOf, dir) => {
        for (let j = lowerBound(list, vt - 2 * D); j < list.length; j++) {
          const f = list[j], s0 = startOf(f);
          if (s0 > vt) { if (f.t > vt + D) break; continue; }
          const u = (vt - s0) / D; if (u < 0 || u > 1) continue;
          if (stride > 1 && (f.h % stride) !== 0) continue;
          const p = this.pos(f.node); if (!p) continue;
          A.set(p.x, p.y + 1.3, p.z);
          srv.set(p.x, 0, p.z).setLength(.8); srv.y = this.sp.coreY;   // from the diamond
          const u2 = Math.min(1, u + .02);
          if (dir > 0) { this._arc(srv, A, ease(u), B); this._arc(srv, A, ease(u2), B2); }
          else { this._arc(A, srv, ease(u), B); this._arc(A, srv, ease(u2), B2); }
          if (B2.distanceToSquared(B) < 1e-8) B2.add(new V3(0, 0, .001));
          tM.lookAt(B2, B, UP); tQ.setFromRotationMatrix(tM);
          tM.compose(B, tQ, tS.setScalar(f.ok === false ? 1.6 : 1.5));
          this.planes.push(tM, tC.setHex(f.ok === false ? 0xe5534b : dir > 0 ? 0x5b9bd5 : 0xf4a259));
        }
      };
      scan(ix.down, f => f.t, +1);
      scan(ix.up, f => f.t - D, -1);
      for (let j = lowerBound(ix.fails, vt - D); j < ix.fails.length; j++) {
        const f = ix.fails[j]; if (f.t > vt) break;
        const p = this.pos(f.node); if (!p) continue;
        const u = (vt - f.t) / D;
        tM.compose(tP.set(p.x, p.y + .02, p.z), tQ.identity(), tS.setScalar(1 + 4 * u));
        this.bursts.push(tM);
      }
    }
    this.planes.end(); this.bursts.end();
  }

  // Aggregation: the books of the clients that replied fly onto the server's pile.
  _ghosts(state, tl, vt, D, sStack) {
    const ix = tl.index;
    this.ghosts.begin();
    if (ix) {
      for (let a = ix.aggs.length - 1; a >= 0; a--) {
        const ag = ix.aggs[a];
        const dur = Math.max(ag.t1 - ag.t0, 1.6 * D);
        if (vt < ag.t0 || vt > ag.t0 + dur) { if (ag.t0 + dur < vt - 10 * D) break; continue; }
        const u = ease(Math.min(1, (vt - ag.t0) / dur));
        const nodes = ag.nodes.length > 30 ? ag.nodes.filter((_, i) => i % Math.ceil(ag.nodes.length / 30) === 0) : ag.nodes;
        for (const id of nodes) {
          const n = state.nodes[id], p = this.pos(id);
          if (!n || !p) continue;
          const yaw = Math.atan2(-p.x, -p.z);
          const right = new V3(Math.cos(yaw), 0, -Math.sin(yaw));
          const sx = p.x + right.x * .8, sz = p.z + right.z * .8;
          const st = this._stack(n.layers.length ? n.layers : sStack.map(s => s.l), .85, 1.0);
          for (let i = 0; i < st.length; i++) {
            const b = st[i], tgt = sStack[Math.min(i, sStack.length - 1)];
            const ty = tgt ? tgt.y : 1;
            const x = sx + (this.serverPile.x - sx) * u, z = sz + (this.serverPile.z - sz) * u;
            const y = (p.y + b.y) * (1 - u) + ty * u + Math.sin(Math.PI * u) * 3.2;
            changeColor(b.l.delta_norm, b.l.norm, tC);
            tQ.setFromAxisAngle(UP, yaw * (1 - u) - .5 * u + u * 6.28);
            tM.compose(tP.set(x, y, z), tQ, tS.set(b.w * (1 - .3 * u), b.h, b.d * (1 - .3 * u)));
            if (this.ghosts.push(tM, tC) < 0) break;
          }
        }
      }
    }
    this.ghosts.end();
  }

  setLens(on) {
    this.lensOn = on;
    const cols = on ? [0xc9c3b8, 0xb3ada2, 0xd8d2c6] : [0x9b6a3c, 0x7a4f2a, 0xe2b34a];
    this.chestMats.forEach((m, i) => { m.color.setHex(cols[i]); m.transparent = on; m.opacity = on ? .45 : 1; m.needsUpdate = true; });
  }

  frame() {
    if (this.mouseMoved) {
      this.mouseMoved = false;
      const h = this._pick();
      const id = h === null ? null : String(h);
      if (id !== this.hover) { this.hover = id; this.renderer.domElement.style.cursor = id ? "pointer" : ""; this.onHover && this.onHover(h); }
    }
    if (this.camGoal) {
      this.controls.target.lerp(this.camGoal.target, .08);
      this.camera.position.lerp(this.camGoal.pos, .08);
      if (this.camera.position.distanceTo(this.camGoal.pos) < .05) this.camGoal = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  project(v) {
    const p = v.clone().project(this.camera);
    const w = this.container.clientWidth, h = this.container.clientHeight;
    return { x: (p.x + 1) / 2 * w, y: (1 - p.y) / 2 * h, visible: p.z < 1 };
  }
}

export function lowerBound(list, t) {
  let lo = 0, hi = list.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (list[m].t < t) lo = m + 1; else hi = m; }
  return lo;
}

// Animation indices built incrementally from the timeline's events.
export class FlightIndex {
  constructor() { this.down = []; this.up = []; this.fails = []; this.aggs = []; this.aggEnds = []; this.n = 0; this.replied = {}; this.open = null; }
  feed(tl) {
    let newFail = false;
    for (; this.n < tl.events.length; this.n++) {
      const e = tl.events[this.n], t = tl.times[this.n], d = e.data || {};
      const h = hash(e.msg_id || String(this.n));
      if (e.type === "msg.sent") this.down.push({ t, node: String(d.dst), h });
      else if (e.type === "msg.reply") {
        this.up.push({ t, node: String(d.from), ok: d.ok, h });
        if (d.ok) (this.replied[`${e.round}|${d.phase}`] ||= []).push(String(d.from));
        else { this.fails.push({ t, node: String(d.from) }); newFail = true; }
      } else if (e.type === "msg.timeout") { this.fails.push({ t, node: String(d.dst) }); newFail = true; }
      else if (e.type === "client.end" && d.ok === false) { this.fails.push({ t, node: String(e.src.node_id) }); newFail = true; }
      else if (e.type === "agg.start" && d.phase === "train") {
        this.open = { t0: t, t1: t, round: e.round, nodes: this.replied[`${e.round}|train`] || [] };
        this.aggs.push(this.open);
      } else if (e.type === "agg.end" && d.phase === "train") {
        if (this.open) this.open.t1 = t;
        this.open = null; this.aggEnds.push(t);
      }
    }
    if (newFail) this.fails.sort((a, b) => a.t - b.t);
  }
}
