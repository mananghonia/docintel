import { useRef, useState } from "react";
import { confBg, confColor } from "../api.js";

/**
 * Document canvas: renders tokens as SVG text, field bounding boxes
 * colour-coded by confidence. Hover links bidirectionally with the field
 * panel. Drag on empty space to draw a box for a field the model missed
 * (a hard negative — the most valuable training example there is).
 */
export default function DocCanvas({ docJson, fields, hovered, onHover, drawTarget, onDrawn }) {
  const svgRef = useRef(null);
  const [drag, setDrag] = useState(null);
  const { page_width: W, page_height: H, tokens = [] } = docJson ?? {};
  if (!W) return <p>No token data.</p>;

  // Multi-page: each page's coordinates are relative to its own page, so stack
  // pages vertically (page p is drawn H*p lower) instead of overlapping them.
  const nPages = Math.max(1, ...tokens.map((t) => (t.page ?? 0) + 1),
    ...(fields ?? []).map((f) => (f.bbox.page ?? 0) + 1));
  const pageY = (p) => (p ?? 0) * H;
  const totalH = nPages * H;

  const toSvgPoint = (e) => {
    const pt = svgRef.current.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(svgRef.current.getScreenCTM().inverse());
  };

  const down = (e) => {
    if (!drawTarget) return;
    const p = toSvgPoint(e);
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const move = (e) => drag && setDrag({ ...drag, ...(() => { const p = toSvgPoint(e); return { x1: p.x, y1: p.y }; })() });
  const up = () => {
    if (drag && drawTarget) {
      // The drag is in stacked-canvas space; map it back to a page-local box.
      const yTop = Math.min(drag.y0, drag.y1);
      const page = Math.max(0, Math.min(nPages - 1, Math.floor(yTop / H)));
      const oy = pageY(page);
      const box = {
        x0: Math.min(drag.x0, drag.x1), y0: Math.min(drag.y0, drag.y1) - oy,
        x1: Math.max(drag.x0, drag.x1), y1: Math.max(drag.y0, drag.y1) - oy, page,
      };
      if (box.x1 - box.x0 > 5 && box.y1 - box.y0 > 5) {
        const inside = tokens.filter((t) => (t.page ?? 0) === page &&
          t.x0 >= box.x0 && t.x1 <= box.x1 && t.y0 >= box.y0 && t.y1 <= box.y1);
        onDrawn(box, inside.map((t) => t.text).join(" "));
      }
    }
    setDrag(null);
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${totalH}`}
      style={{ width: "100%", display: "block", cursor: drawTarget ? "crosshair" : "default" }}
      onMouseDown={down} onMouseMove={move} onMouseUp={up}
    >
      {Array.from({ length: nPages }, (_, p) => (
        <rect key={`pg${p}`} x={0} y={pageY(p)} width={W} height={H}
          fill="#fff" stroke="#e2e8f0" />
      ))}
      {tokens.map((t, i) => {
        // Scale each glyph to its OCR bounding box instead of a fixed size:
        // a fixed 16px font turns a dense page (1000+ small tokens) into an
        // overlapping smear. textLength + spacingAndGlyphs squeezes each word
        // to exactly its box width so nothing bleeds into its neighbour.
        const w = Math.max(t.x1 - t.x0, 1);
        const h = Math.max(t.y1 - t.y0, 1);
        const oy = pageY(t.page);
        return (
          <text
            key={i}
            x={t.x0}
            y={oy + t.y1 - h * 0.18}
            fontSize={Math.min(Math.max(h, 4), 40)}
            textLength={w}
            lengthAdjust="spacingAndGlyphs"
            fontFamily="sans-serif"
            fill="#1e293b"
          >
            {t.text}
          </text>
        );
      })}
      {(fields ?? []).map((f) => {
        const oy = pageY(f.bbox.page);
        return (
          <rect
            key={f.field}
            x={f.bbox.x0 - 3} y={oy + f.bbox.y0 - 3}
            width={f.bbox.x1 - f.bbox.x0 + 6} height={f.bbox.y1 - f.bbox.y0 + 6}
            fill={confBg(f.confidence)}
            stroke={confColor(f.confidence)}
            strokeWidth={hovered === f.field ? 4 : 1.5}
            onMouseEnter={() => onHover(f.field)}
            onMouseLeave={() => onHover(null)}
          />
        );
      })}
      {drag && (
        <rect
          x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)}
          width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)}
          fill="rgba(14,165,233,.15)" stroke="#0ea5e9" strokeDasharray="6 3"
        />
      )}
    </svg>
  );
}
