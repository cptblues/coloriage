"use client";

import {
  ChangeEvent,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

type DetailZone = {
  mode: "paint" | "outline";
  points: Array<{ x: number; y: number }>;
  radius: number;
};

type ImagePayload = {
  name: string;
  type: string;
  dataUrl: string;
};

type MaskStatus = "idle" | "loading" | "ready" | "error";

type EnginePaletteEntry = {
  number: number;
  hex: string;
  rgb: number[];
};

type EngineResult = {
  actualColors: number;
  regionsAfter: number;
  palette: EnginePaletteEntry[];
  coloringPreviewImage: string;
  modelPreviewImage: string;
  palettePageImage?: string | null;
  pdfDocument?: string | null;
  maskControlImage?: string | null;
  detailMaskImage?: string | null;
  stats?: {
    result?: {
      labeling?: {
        placed_count?: number;
        skipped_count?: number;
        coverage_percent?: number;
        reduced_font_count?: number;
      };
    };
  };
};

type FittedBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

const STEPS = [
  { short: "Upload", long: "Choisir une photo" },
  { short: "Isolation", long: "Contrôler le fond" },
  { short: "Zones détaillées", long: "Préserver les détails" },
  { short: "Réglages", long: "Ajuster le rendu" },
  { short: "Résultat", long: "Votre coloriage" },
];

const PALETTE = ["#8fa18a", "#f0bb5d", "#7ea3bd", "#a793b3", "#db6f5a", "#243750"];
const DEFAULT_DETAIL_BRUSH_SIZE = 42;
const MIN_DETAIL_POINT_DISTANCE = 0.0015;
const ENGINE_URL = (
  import.meta.env.VITE_ENGINE_URL || "http://127.0.0.1:8765"
).replace(/\/+$/, "");

type PointerSample = {
  clientX: number;
  clientY: number;
};

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function canvasDisplaySize(canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  return {
    width: Math.max(1, rect.width || canvas.clientWidth || canvas.width),
    height: Math.max(1, rect.height || canvas.clientHeight || canvas.height),
  };
}

function prepareCanvas(canvas: HTMLCanvasElement) {
  const { width, height } = canvasDisplaySize(canvas);
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(width * dpr));
  const pixelHeight = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function pointerSamples(event: ReactPointerEvent<HTMLCanvasElement>): PointerSample[] {
  const nativeEvent = event.nativeEvent as PointerEvent & {
    getCoalescedEvents?: () => PointerEvent[];
  };
  const coalesced = nativeEvent.getCoalescedEvents?.();
  return coalesced?.length ? coalesced : [event];
}

function drawDetailZonePath(
  ctx: CanvasRenderingContext2D,
  zone: DetailZone,
  box: FittedBox,
  closePath = false,
) {
  const minSize = Math.min(box.w, box.h);
  const radius = zone.radius * minSize;
  const points = zone.points.map((point) => ({
    x: box.x + point.x * box.w,
    y: box.y + point.y * box.h,
  }));

  if (points.length === 0) return;
  ctx.beginPath();
  if (points.length <= 1) {
    const point = points[0];
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    return;
  }

  if (closePath && points.length >= 3) {
    const firstMidpoint = {
      x: (points[0].x + points[1].x) / 2,
      y: (points[0].y + points[1].y) / 2,
    };
    ctx.moveTo(firstMidpoint.x, firstMidpoint.y);
    for (let index = 1; index <= points.length; index += 1) {
      const control = points[index % points.length];
      const next = points[(index + 1) % points.length];
      ctx.quadraticCurveTo(
        control.x,
        control.y,
        (control.x + next.x) / 2,
        (control.y + next.y) / 2,
      );
    }
    ctx.closePath();
    return;
  }

  ctx.moveTo(points[0].x, points[0].y);
  if (points.length === 2) {
    ctx.lineTo(points[1].x, points[1].y);
  } else {
    for (let index = 1; index < points.length - 1; index += 1) {
      const current = points[index];
      const next = points[index + 1];
      ctx.quadraticCurveTo(
        current.x,
        current.y,
        (current.x + next.x) / 2,
        (current.y + next.y) / 2,
      );
    }
    const last = points[points.length - 1];
    ctx.lineTo(last.x, last.y);
  }
}

function imageBox(
  image: HTMLImageElement,
  width: number,
  height: number,
): FittedBox {
  const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const w = image.naturalWidth * scale;
  const h = image.naturalHeight * scale;
  const x = (width - w) / 2;
  const y = (height - h) / 2;
  return { x, y, w, h };
}

function fitImage(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
) {
  const box = imageBox(image, width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f5f0e4";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, box.x, box.y, box.w, box.h);
  return box;
}

function BrandMark() {
  return (
    <a className="brand" href="#" aria-label="Nuance, accueil">
      <span className="brand-fan" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
        <i />
      </span>
      <span>Nuance</span>
      <b aria-hidden="true">✦</b>
    </a>
  );
}

function Progress({ active }: { active: number }) {
  return (
    <nav className="progress" aria-label="Étapes de création">
      {STEPS.map((item, index) => (
        <div
          className={`progress-item ${index === active ? "active" : ""} ${index < active ? "done" : ""}`}
          key={item.short}
          aria-current={index === active ? "step" : undefined}
        >
          <span>{index < active ? "✓" : index + 1}</span>
          <strong>{item.short}</strong>
        </div>
      ))}
    </nav>
  );
}

function SecurityNote() {
  return (
    <div className="security-note">
      <span aria-hidden="true">●</span>
      Votre photo reste sur cet appareil.
    </div>
  );
}

export default function Home() {
  const [step, setStep] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imagePayload, setImagePayload] = useState<ImagePayload | null>(null);
  const [fileName, setFileName] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [maskStatus, setMaskStatus] = useState<MaskStatus>("idle");
  const [maskError, setMaskError] = useState("");
  const [maskNotice, setMaskNotice] = useState("");
  const [maskPreviewImage, setMaskPreviewImage] = useState<string | null>(null);
  const [subjectMaskImage, setSubjectMaskImage] = useState<string | null>(null);
  const [zones, setZones] = useState<DetailZone[]>([]);
  const [draftZone, setDraftZone] = useState<DetailZone | null>(null);
  const [detailMode, setDetailMode] = useState<"paint" | "outline">("paint");
  const [detailBrushSize, setDetailBrushSize] = useState(DEFAULT_DETAIL_BRUSH_SIZE);
  const [colors, setColors] = useState(24);
  const [complexity, setComplexity] = useState<"simple" | "equilibre" | "detaille">(
    "equilibre",
  );
  const [format, setFormat] = useState<"A4" | "A3">("A4");
  const [orientation, setOrientation] = useState<"portrait" | "paysage">("portrait");
  const [title, setTitle] = useState("Mon coloriage mystère");
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState("");
  const [engineResult, setEngineResult] = useState<EngineResult | null>(null);
  const [resultMode, setResultMode] = useState<"color" | "line" | "palette">("line");
  const fileInput = useRef<HTMLInputElement>(null);
  const detailCanvas = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const detailDrawing = useRef(false);
  const detailDraftRef = useRef<DetailZone | null>(null);

  const loadFile = useCallback((file?: File) => {
    if (!file || !["image/jpeg", "image/png"].includes(file.type)) return;
    const nextUrl = URL.createObjectURL(file);
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") return;
      setImagePayload({
        name: file.name,
        type: file.type,
        dataUrl: reader.result,
      });
    };
    reader.readAsDataURL(file);
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return nextUrl;
    });
    setFileName(file.name);
    setMaskStatus("idle");
    setMaskError("");
    setMaskNotice("");
    setMaskPreviewImage(null);
    setSubjectMaskImage(null);
    setZones([]);
    setEngineResult(null);
    setGenerationError("");
  }, []);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    loadFile(event.target.files?.[0]);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    loadFile(event.dataTransfer.files?.[0]);
  };

  useEffect(() => {
    if (!imageUrl) return;
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      [detailCanvas.current].forEach((canvas) => {
        if (!canvas) return;
        const prepared = prepareCanvas(canvas);
        if (!prepared) return;
        fitImage(prepared.ctx, image, prepared.width, prepared.height);
      });
    };
    image.src = imageUrl;
  }, [imageUrl, step]);

  const requestSubjectMask = useCallback(async () => {
    if (!imagePayload || maskStatus === "loading") return;
    setMaskStatus("loading");
    setMaskError("");
    setMaskNotice("");
    try {
      const response = await fetch(`${ENGINE_URL}/mask`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          image: imagePayload,
          maxSide: "auto",
        }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Échec de l'isolation du fond.");
      }
      const nextSubjectMask =
        typeof payload.subjectMaskImage === "string" ? payload.subjectMaskImage : null;
      setMaskPreviewImage(payload.maskControlImage || imageUrl);
      setSubjectMaskImage(nextSubjectMask);
      setMaskNotice(
        typeof payload.subject?.message === "string"
          ? payload.subject.message
          : nextSubjectMask
            ? "Sujet détecté : le fond sera simplifié plus fortement."
            : "Aucun sujet net détecté : génération globale.",
      );
      setMaskStatus("ready");
    } catch (error) {
      setMaskStatus("error");
      setMaskError(
        error instanceof Error
          ? error.message
          : "Le moteur Python local n'a pas répondu.",
      );
    }
  }, [imagePayload, imageUrl, maskStatus]);

  useEffect(() => {
    if (step === 1 && imagePayload && maskStatus === "idle") {
      const timer = window.setTimeout(() => {
        void requestSubjectMask();
      }, 0);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [imagePayload, maskStatus, requestSubjectMask, step]);

  const renderDetailCanvas = useCallback(() => {
    const canvas = detailCanvas.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const prepared = prepareCanvas(canvas);
    if (!prepared) return;
    const { ctx, width, height } = prepared;
    const box = fitImage(ctx, image, width, height);
    const drawZone = (zone: DetailZone, label?: number, isDraft = false) => {
      const minSize = Math.min(box.w, box.h);
      const radius = zone.radius * minSize;
      const labelPoint = zone.points[Math.floor(zone.points.length / 2)] ?? zone.points[0];
      const labelX = box.x + labelPoint.x * box.w;
      const labelY = box.y + labelPoint.y * box.h;

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      if (zone.mode === "outline" && zone.points.length >= 3) {
        drawDetailZonePath(ctx, zone, box, true);
        ctx.fillStyle = isDraft ? "rgba(239, 184, 78, .34)" : "rgba(239, 184, 78, .26)";
        ctx.fill();
        drawDetailZonePath(ctx, zone, box, true);
        ctx.strokeStyle = "rgba(255, 255, 255, .96)";
        ctx.lineWidth = Math.max(8, radius * 0.56);
        ctx.stroke();
        drawDetailZonePath(ctx, zone, box, true);
        ctx.strokeStyle = isDraft ? "#f5b843" : "#c53f24";
        ctx.lineWidth = Math.max(4, radius * 0.28);
        ctx.stroke();
      } else {
        drawDetailZonePath(ctx, zone, box);
        ctx.strokeStyle = isDraft ? "rgba(239, 184, 78, .42)" : "rgba(239, 184, 78, .34)";
        ctx.lineWidth = radius * 2.4;
        ctx.stroke();
        drawDetailZonePath(ctx, zone, box);
        ctx.strokeStyle = "rgba(255, 255, 255, .95)";
        ctx.lineWidth = Math.max(10, radius * 0.72);
        ctx.stroke();
        drawDetailZonePath(ctx, zone, box);
        ctx.strokeStyle = isDraft ? "#f5b843" : "#c53f24";
        ctx.lineWidth = Math.max(5, radius * 0.34);
        ctx.stroke();
      }
      ctx.restore();

      if (label) {
        ctx.save();
        ctx.fillStyle = "#c53f24";
        ctx.beginPath();
        ctx.arc(labelX, labelY, 19, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fffdf7";
        ctx.font = "800 18px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(label), labelX, labelY + 1);
        ctx.restore();
      }
    };

    zones.forEach((zone, index) => {
      drawZone(zone, index + 1);
    });
    if (draftZone) drawZone(draftZone, undefined, true);
  }, [draftZone, zones]);

  useEffect(() => {
    if (step !== 2) return undefined;
    renderDetailCanvas();
    const canvas = detailCanvas.current;
    if (!canvas || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => renderDetailCanvas());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [renderDetailCanvas, step]);

  const getDetailPoint = (canvas: HTMLCanvasElement, sample: PointerSample) => {
    const image = imageRef.current;
    const rect = canvas.getBoundingClientRect();
    const { width, height } = canvasDisplaySize(canvas);
    const canvasX = sample.clientX - rect.left;
    const canvasY = sample.clientY - rect.top;
    const box = image
      ? imageBox(image, width, height)
      : { x: 0, y: 0, w: width, h: height };
    return {
      x: clamp01((canvasX - box.x) / box.w),
      y: clamp01((canvasY - box.y) / box.h),
    };
  };

  const getDetailRadius = (canvas: HTMLCanvasElement) => {
    const image = imageRef.current;
    const { width, height } = canvasDisplaySize(canvas);
    const box = image
      ? imageBox(image, width, height)
      : { x: 0, y: 0, w: width, h: height };
    return Math.max(0.006, detailBrushSize / (2.4 * Math.min(box.w, box.h)));
  };

  const appendDetailSamples = (
    canvas: HTMLCanvasElement,
    event: ReactPointerEvent<HTMLCanvasElement>,
    updateDraft = true,
  ) => {
    const current = detailDraftRef.current;
    if (!current) return null;
    const points = [...current.points];
    let changed = false;
    for (const sample of pointerSamples(event)) {
      const point = getDetailPoint(canvas, sample);
      const lastPoint = points[points.length - 1];
      if (
        !lastPoint ||
        Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) >=
          MIN_DETAIL_POINT_DISTANCE
      ) {
        points.push(point);
        changed = true;
      }
    }
    if (!changed) return current;
    const nextZone = { ...current, points };
    detailDraftRef.current = nextZone;
    if (updateDraft) setDraftZone(nextZone);
    return nextZone;
  };

  const startDetailZone = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    detailDrawing.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = getDetailPoint(event.currentTarget, event);
    const nextZone = {
      mode: detailMode,
      points: [point],
      radius: getDetailRadius(event.currentTarget),
    };
    detailDraftRef.current = nextZone;
    setDraftZone(nextZone);
  };

  const resizeDetailZone = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!detailDrawing.current) return;
    event.preventDefault();
    appendDetailSamples(event.currentTarget, event);
  };

  const finishDetailZone = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!detailDrawing.current) return;
    event.preventDefault();
    detailDrawing.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const completedZone =
      appendDetailSamples(event.currentTarget, event, false) ?? detailDraftRef.current;
    detailDraftRef.current = null;
    setDraftZone(null);
    if (completedZone) setZones((existing) => [...existing, completedZone]);
  };

  const cancelDetailZone = () => {
    detailDrawing.current = false;
    detailDraftRef.current = null;
    setDraftZone(null);
  };

  const generate = async () => {
    if (!imagePayload) return;
    setGenerating(true);
    setGenerationError("");
    try {
      const response = await fetch(`${ENGINE_URL}/generate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          image: imagePayload,
          subjectMaskImage,
          detailZones: zones,
          colors,
          complexity,
          format,
          orientation,
          title,
          paletteLayout: "separate",
          maxSide: "auto",
        }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Échec de la génération.");
      }
      setEngineResult(payload);
      setStep(4);
    } catch (error) {
      setGenerationError(
        error instanceof Error
          ? error.message
          : "Le moteur Python local n'a pas répondu.",
      );
    } finally {
      setGenerating(false);
    }
  };

  const downloadResult = () => {
    if (!engineResult) return;
    const href =
      resultMode === "line"
        ? engineResult.coloringPreviewImage
        : resultMode === "color"
          ? engineResult.modelPreviewImage
          : engineResult.palettePageImage;
    if (!href) return;
    const link = document.createElement("a");
    link.download = `nuance-${resultMode}.png`;
    link.href = href;
    link.click();
  };

  const downloadPdf = () => {
    if (!engineResult?.pdfDocument) return;
    const link = document.createElement("a");
    link.download = "nuance-coloriage.pdf";
    link.href = engineResult.pdfDocument;
    link.click();
  };

  const goNext = () => {
    if (step === 0 && !imageUrl) {
      fileInput.current?.click();
      return;
    }
    if (step === 1 && maskStatus !== "ready") {
      void requestSubjectMask();
      return;
    }
    if (step === 3) {
      void generate();
      return;
    }
    setStep((current) => Math.min(current + 1, 4));
  };

  return (
    <main>
      <header className="topbar">
        <BrandMark />
        <Progress active={step} />
        <button className="quiet-button" type="button" onClick={() => setStep(0)}>
          Recommencer
        </button>
      </header>

      <div className="mobile-progress">
        Étape {step + 1} sur 5 · <strong>{STEPS[step].long}</strong>
      </div>

      <section className="workspace">
        {step === 0 && (
          <>
            <article className="control-card intro-card">
              <p className="eyebrow">Votre photo, votre création</p>
              <h1>
                Transformez une photo
                <br />
                en <span>coloriage mystère</span>
              </h1>
              <p className="lead">
                Importez un souvenir, isolez le sujet et choisissez où conserver les
                plus jolies nuances.
              </p>
              <div
                className={`dropzone ${isDragging ? "dragging" : ""} ${imageUrl ? "has-file" : ""}`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInput.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") fileInput.current?.click();
                }}
              >
                <input
                  accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                  onChange={onFileChange}
                  ref={fileInput}
                  type="file"
                />
                <span className="upload-icon" aria-hidden="true">
                  {imageUrl ? "✓" : "↑"}
                </span>
                <strong>{imageUrl ? fileName : "Glissez votre photo ici"}</strong>
                <small>{imageUrl ? "Prête à être transformée" : "ou cliquez pour parcourir"}</small>
                <div className="file-pills">
                  <span>JPG</span>
                  <span>PNG</span>
                  <span>40 Mo max.</span>
                </div>
              </div>
              <SecurityNote />
              <button className="primary-button" type="button" onClick={goNext}>
                {imageUrl ? "Commencer la création" : "Choisir une photo"}
                <span aria-hidden="true">↗</span>
              </button>
            </article>

            <article className="hero-art">
              <img
                src="/nuance-hero.png"
                alt="Exemple de coloriage mystère représentant une femme et son chat"
              />
              <div className="paper-label">Un souvenir à colorier</div>
              <div className="palette-strip" aria-label="Exemple de palette">
                {PALETTE.map((color, index) => (
                  <span key={color}>
                    <i style={{ backgroundColor: color }} />
                    {index + 1}
                  </span>
                ))}
              </div>
            </article>
          </>
        )}

        {step === 1 && (
          <>
            <article className="control-card step-card">
              <p className="step-kicker">Étape 2 · Isolation du fond</p>
              <h1>Vérifions le détourage.</h1>
              <p className="lead">
                Si un sujet net est détecté, le moteur simplifie le fond plus
                fortement. Pour un paysage, il peut aussi générer l’image
                globalement.
              </p>
              <div className={`engine-status ${maskStatus}`}>
                <span className={`status-dot ${maskStatus === "ready" ? "ready" : ""}`} />
                <div>
                  <b>
                    {maskStatus === "loading"
                      ? "Isolation en cours"
                      : maskStatus === "ready"
                        ? subjectMaskImage
                          ? "Fond isolé"
                          : "Traitement global"
                        : maskStatus === "error"
                          ? "Moteur indisponible"
                          : "En attente du moteur"}
                  </b>
                  <small>
                    {maskStatus === "error"
                      ? maskError
                      : maskStatus === "ready"
                        ? maskNotice
                        : `Service attendu sur ${ENGINE_URL}`}
                  </small>
                </div>
              </div>
              <div className="tip">
                <b>Astuce</b>
                Vous n’avez rien à peindre ici. À l’étape suivante, tracez
                seulement les zones où conserver plus de nuances.
              </div>
              <button
                className="primary-button"
                disabled={maskStatus === "loading"}
                type="button"
                onClick={goNext}
              >
                {maskStatus === "ready"
                  ? "Continuer"
                  : maskStatus === "loading"
                    ? "Isolation en cours…"
                    : "Lancer l’isolation"}
                <span aria-hidden="true">→</span>
              </button>
              {maskStatus === "error" && (
                <button className="secondary-button" type="button" onClick={requestSubjectMask}>
                  Réessayer
                </button>
              )}
            </article>
            <article className="canvas-card">
              <div className="canvas-heading">
                <div>
                  <span className={`status-dot ${maskStatus === "ready" ? "ready" : ""}`} />
                  {maskStatus === "ready" && subjectMaskImage
                    ? "Contrôle du masque"
                    : maskStatus === "ready"
                      ? "Traitement global"
                      : "Photo importée"}
                </div>
                {maskStatus === "ready" && (
                  <button type="button" onClick={requestSubjectMask}>
                    Relancer
                  </button>
                )}
              </div>
              <div className="preview-frame">
                {maskPreviewImage || imageUrl ? (
                  <img
                    src={maskPreviewImage || imageUrl || ""}
                    alt="Contrôle de l’isolation du fond"
                  />
                ) : (
                  <span>Importez une photo pour lancer l’isolation.</span>
                )}
                {maskStatus === "loading" && (
                  <div className="preview-overlay">Isolation locale en cours…</div>
                )}
              </div>
            </article>
          </>
        )}

        {step === 2 && (
          <>
            <article className="control-card step-card">
              <p className="step-kicker">Étape 3 · Zones détaillées</p>
              <h1>Où garder plus de nuances ?</h1>
              <p className="lead">
                Peignez directement les détails à préserver, ou dessinez un contour
                que Nuance remplira au relâchement.
              </p>
              <fieldset>
                <legend>Mode de sélection</legend>
                <div className="segmented">
                  <button
                    className={detailMode === "paint" ? "selected" : ""}
                    type="button"
                    onClick={() => setDetailMode("paint")}
                  >
                    Pinceau
                  </button>
                  <button
                    className={detailMode === "outline" ? "selected" : ""}
                    type="button"
                    onClick={() => setDetailMode("outline")}
                  >
                    Contour
                  </button>
                </div>
              </fieldset>
              <label className="range-label detail-brush">
                Épaisseur <b>{detailBrushSize}px</b>
                <input
                  max="96"
                  min="12"
                  type="range"
                  value={detailBrushSize}
                  onChange={(event) => setDetailBrushSize(Number(event.target.value))}
                />
              </label>
              <div className="gesture-help">
                <span aria-hidden="true">{detailMode === "paint" ? "●" : "◎"}</span>
                <div>
                  <b>{detailMode === "paint" ? "Peignez la zone" : "Dessinez le contour"}</b>
                  {detailMode === "paint"
                    ? "Relâchez pour enregistrer ce que vous avez peint."
                    : "Relâchez pour fermer et remplir la zone entourée."}
                </div>
              </div>
              <div className="zone-list">
                {zones.length === 0 ? (
                  <p>Aucune zone ajoutée pour le moment.</p>
                ) : (
                  zones.map((zone, index) => (
                    <div key={index}>
                      <span>{index + 1}</span>
                      {zone.mode === "outline" ? "Contour rempli" : "Zone peinte"} {index + 1}
                      <button
                        aria-label={`Supprimer la zone ${index + 1}`}
                        type="button"
                        onClick={() =>
                          setZones((current) => current.filter((__, i) => i !== index))
                        }
                      >
                        ×
                      </button>
                    </div>
                  ))
                )}
              </div>
              <div className="tip">
                <b>Bon à savoir</b>
                Les contours des zones sont progressifs : aucun tracé ne sera visible
                dans le résultat.
              </div>
              <button className="primary-button" type="button" onClick={goNext}>
                Continuer avec {zones.length} zone{zones.length > 1 ? "s" : ""}
                <span aria-hidden="true">→</span>
              </button>
            </article>
            <article className="canvas-card detail-canvas">
              <div className="canvas-heading">
                <div>
                  <span className="status-dot ready" />
                  {detailMode === "paint" ? "Peignez une zone" : "Tracez un contour"}
                </div>
                <div className="canvas-actions">
                  <button
                    disabled={zones.length === 0}
                    type="button"
                    onClick={() => setZones((current) => current.slice(0, -1))}
                  >
                    Annuler
                  </button>
                  <button disabled={zones.length === 0} type="button" onClick={() => setZones([])}>
                    Tout effacer
                  </button>
                </div>
              </div>
              <div className="detail-canvas-frame">
                <canvas
                  aria-label="Sélection des zones détaillées"
                  onPointerCancel={cancelDetailZone}
                  onPointerDown={startDetailZone}
                  onPointerMove={resizeDetailZone}
                  onPointerUp={finishDetailZone}
                  ref={detailCanvas}
                />
              </div>
              {zones.length === 0 && !draftZone && (
                <div className="canvas-hint">
                  <span>{detailMode === "paint" ? "●" : "◎"}</span>
                  <div>
                    <b>Tracez votre première zone</b>
                    {detailMode === "paint"
                      ? "Peignez les détails à garder"
                      : "Entourez la zone, puis relâchez"}
                  </div>
                </div>
              )}
            </article>
          </>
        )}

        {step === 3 && (
          <>
            <article className="control-card step-card settings-card">
              <p className="step-kicker">Étape 4 · Réglages</p>
              <h1>À vous de doser le défi.</h1>
              <p className="lead">
                Ajustez la palette et la complexité. Vous pourrez revenir ici après
                la génération.
              </p>
              <label className="setting-field">
                <span>
                  Nombre de couleurs <b>{colors}</b>
                </span>
                <input
                  max="40"
                  min="8"
                  type="range"
                  value={colors}
                  onChange={(event) => setColors(Number(event.target.value))}
                />
                <small>Simple</small>
                <small>Nuancé</small>
              </label>
              <fieldset>
                <legend>Niveau de détail</legend>
                <div className="segmented">
                  {(["simple", "equilibre", "detaille"] as const).map((value) => (
                    <button
                      className={complexity === value ? "selected" : ""}
                      key={value}
                      type="button"
                      onClick={() => setComplexity(value)}
                    >
                      {value === "simple"
                        ? "Simple"
                        : value === "equilibre"
                          ? "Équilibré"
                          : "Détaillé"}
                    </button>
                  ))}
                </div>
              </fieldset>
              <div className="format-row">
                <fieldset>
                  <legend>Format</legend>
                  <div className="segmented">
                    {(["A4", "A3"] as const).map((value) => (
                      <button
                        className={format === value ? "selected" : ""}
                        key={value}
                        type="button"
                        onClick={() => setFormat(value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <fieldset>
                  <legend>Orientation</legend>
                  <div className="segmented">
                    {(["portrait", "paysage"] as const).map((value) => (
                      <button
                        className={orientation === value ? "selected" : ""}
                        key={value}
                        type="button"
                        onClick={() => setOrientation(value)}
                      >
                        {value === "portrait" ? "Verticale" : "Horizontale"}
                      </button>
                    ))}
                  </div>
                </fieldset>
              </div>
              <label className="text-field">
                Titre du coloriage
                <input value={title} onChange={(event) => setTitle(event.target.value)} />
              </label>
              <button className="primary-button" disabled={generating} type="button" onClick={generate}>
                {generating ? "Création Python en cours…" : "Créer mon coloriage"}
                <span aria-hidden="true">{generating ? "◌" : "✦"}</span>
              </button>
              {generationError && <p className="error-message">{generationError}</p>}
            </article>
            <article className="summary-board">
              <div className="summary-paper">
                <p>Aperçu de vos choix</p>
                <h2>{title || "Mon coloriage mystère"}</h2>
                <div className="summary-photo">
                  {imageUrl && <img src={imageUrl} alt="" />}
                  <span>{colors} couleurs</span>
                </div>
                <dl>
                  <div>
                    <dt>Sujet</dt>
                    <dd>
                      {subjectMaskImage
                        ? "isolé"
                        : maskStatus === "ready"
                          ? "global"
                          : "à isoler"}
                    </dd>
                  </div>
                  <div>
                    <dt>Zones détaillées</dt>
                    <dd>{zones.length}</dd>
                  </div>
                  <div>
                    <dt>Sortie</dt>
                    <dd>
                      {format} · {orientation === "portrait" ? "verticale" : "horizontale"}
                    </dd>
                  </div>
                </dl>
              </div>
            </article>
          </>
        )}

        {step === 4 && (
          <>
            <article className="control-card step-card result-controls">
              <p className="step-kicker">Étape 5 · Résultat</p>
              <h1>Votre coloriage prend vie.</h1>
              <p className="lead">
                Comparez la feuille numérotée, le modèle coloré et la palette
                séparée, puis téléchargez la page souhaitée.
              </p>
              <div className="result-tabs">
                <button
                  className={resultMode === "line" ? "selected" : ""}
                  type="button"
                  onClick={() => setResultMode("line")}
                >
                  Coloriage
                </button>
                <button
                  className={resultMode === "color" ? "selected" : ""}
                  type="button"
                  onClick={() => setResultMode("color")}
                >
                  Modèle couleur
                </button>
                <button
                  className={resultMode === "palette" ? "selected" : ""}
                  disabled={!engineResult?.palettePageImage}
                  type="button"
                  onClick={() => setResultMode("palette")}
                >
                  Palette
                </button>
              </div>
              <div className="result-stats">
                <div>
                  <strong>{engineResult?.actualColors ?? colors}</strong>
                  couleurs
                </div>
                <div>
                  <strong>{engineResult?.regionsAfter ?? "—"}</strong>
                  zones
                </div>
                <div>
                  <strong>
                    {engineResult?.stats?.result?.labeling?.placed_count ?? "—"} /{" "}
                    {engineResult?.regionsAfter ?? "—"}
                  </strong>
                  zones numérotées
                </div>
              </div>
              <button
                className="primary-button"
                disabled={!engineResult?.pdfDocument}
                type="button"
                onClick={downloadPdf}
              >
                Télécharger le PDF
                <span aria-hidden="true">↓</span>
              </button>
              <button
                className="secondary-button"
                disabled={!engineResult || (resultMode === "palette" && !engineResult.palettePageImage)}
                type="button"
                onClick={downloadResult}
              >
                Télécharger cette page PNG
              </button>
              <button className="secondary-button" type="button" onClick={() => setStep(3)}>
                Ajuster les réglages
              </button>
              <p className="demo-note">
                Aperçu généré par le moteur Python local. Les numéros correspondent
                aux entrées de palette affichées.
              </p>
            </article>
            <article className="result-board">
              <div className="result-paper">
                <div className="result-title">
                  <span>NUANCE</span>
                  <strong>{title}</strong>
                  <em>
                    {format} · {engineResult?.actualColors ?? colors} couleurs
                  </em>
                </div>
                {engineResult ? (
                  <img
                    className="result-preview-image"
                    src={
                      resultMode === "line"
                        ? engineResult.coloringPreviewImage
                        : resultMode === "color"
                          ? engineResult.modelPreviewImage
                          : engineResult.palettePageImage || engineResult.modelPreviewImage
                    }
                    alt={
                      resultMode === "line"
                        ? "Aperçu du coloriage numéroté"
                        : resultMode === "color"
                          ? "Aperçu du modèle coloré"
                          : "Page palette du coloriage"
                    }
                  />
                ) : (
                  <div className="result-empty">Aucun résultat généré.</div>
                )}
              </div>
            </article>
          </>
        )}
      </section>

      <footer>
        <SecurityNote />
        <span>POC Lot 4 · Traitement local dans le navigateur</span>
      </footer>
    </main>
  );
}
