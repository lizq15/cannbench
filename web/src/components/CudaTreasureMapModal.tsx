import { useEffect } from "react";
import { createPortal } from "react-dom";
import { cudaTreasureRoute } from "../data/cudaOptimizationRoute";
import { CudaTreasureMap } from "./CudaTreasureMap";

interface CudaTreasureMapModalProps {
  open: boolean;
  onClose: () => void;
}

export function CudaTreasureMapModal({ open, onClose }: CudaTreasureMapModalProps) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div className="modal-backdrop cuda-treasure-map-modal__backdrop" role="presentation" onClick={onClose}>
      <section
        className="cuda-treasure-map-modal"
        role="dialog"
        aria-modal="true"
        aria-label="CUDA operator treasure route"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="cuda-treasure-map-modal__header">
          <div>
            <p className="panel-kicker">CUDA treasure map</p>
            <h2>CUDA operator treasure route</h2>
          </div>
          <button type="button" className="modal-close" aria-label="Close CUDA operator treasure route" onClick={onClose}>
            close
          </button>
        </header>
        <CudaTreasureMap route={cudaTreasureRoute} />
      </section>
    </div>,
    document.body
  );
}
