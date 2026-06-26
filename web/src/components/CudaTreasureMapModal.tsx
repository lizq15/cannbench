import { cudaTreasureRoute } from "../data/cudaOptimizationRoute";

interface CudaTreasureMapModalProps {
  open: boolean;
  onClose: () => void;
}

export function CudaTreasureMapModal({ open, onClose: _onClose }: CudaTreasureMapModalProps) {
  if (!open) {
    return null;
  }

  const mainRouteNodes = cudaTreasureRoute.filter((node) => node.kind === "main");

  return (
    <section role="dialog" aria-modal="true" aria-label="CUDA operator treasure route">
      {mainRouteNodes.map((node) => (
        <p key={node.id}>{node.label}</p>
      ))}
    </section>
  );
}
