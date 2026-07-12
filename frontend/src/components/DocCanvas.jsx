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
      const box = {
        x0: Math.min(drag.x0, drag.x1), y0: Math.min(drag.y0, drag.y1),
        x1: Math.max(drag.x0, drag.x1), y1: Math.max(drag.y0, drag.y1), page: 0,
      };
      if (box.x1 - box.x0 > 5 && box.y1 - box.y0 > 5) {
        const inside = tokens.filter((t) => t.x0 >= box.x0 && t.x1 <= box.x1 && t.y0 >= box.y0 && t.y1 <= box.y1);
        onDrawn(box, inside.map((t) => t.text).join(" "));
      }
    }
    setDrag(null);
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", display: "block", cursor: drawTarget ? "crosshair" : "default" }}
      onMouseDown={down} onMouseMove={move} onMouseUp={up}
    >
      <rect width={W} height={H} fill="#fff" />
      {tokens.map((t, i) => (
        <text key={i} x={t.x0} y={t.y1} fontSize={16} fontFamily="monospace" fill="#1e293b">
          {t.text}
        </text>
      ))}
      {(fields ?? []).map((f) => (
        <rect
          key={f.field}
          x={f.bbox.x0 - 3} y={f.bbox.y0 - 3}
          width={f.bbox.x1 - f.bbox.x0 + 6} height={f.bbox.y1 - f.bbox.y0 + 6}
          fill={confBg(f.confidence)}
          stroke={confColor(f.confidence)}
          strokeWidth={hovered === f.field ? 4 : 1.5}
          onMouseEnter={() => onHover(f.field)}
          onMouseLeave={() => onHover(null)}
        />
      ))}
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
