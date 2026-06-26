import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CudaTreasureMapModal } from "./CudaTreasureMapModal";

afterEach(() => {
  cleanup();
});

describe("CudaTreasureMapModal", () => {
  it("renders the main CUDA optimization route labels", () => {
    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    expect(screen.getByRole("dialog", { name: /CUDA operator treasure route/i })).toBeInTheDocument();
    expect(screen.getByText(/Profile the Truth/i)).toBeInTheDocument();
    expect(screen.getByText(/Fix Global Access/i)).toBeInTheDocument();
    expect(screen.getByText(/Polish Instructions/i)).toBeInTheDocument();
  });

  it("reveals the Fix Global Access field note on hover and focus", async () => {
    const user = userEvent.setup();

    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    const fixGlobalAccessNode = screen.getByRole("button", { name: /Fix Global Access/i });

    expect(screen.queryByText(/^Guide:\s*10\.2\.1$/i)).not.toBeInTheDocument();

    await user.hover(fixGlobalAccessNode);
    expect(screen.getByText(/^Guide:\s*10\.2\.1$/i)).toBeInTheDocument();
    expect(screen.getByText(/Repair coalescing, stride, and alignment before instruction micro-tuning\./i)).toBeInTheDocument();

    await user.unhover(fixGlobalAccessNode);
    expect(screen.queryByText(/^Guide:\s*10\.2\.1$/i)).not.toBeInTheDocument();

    fixGlobalAccessNode.focus();
    expect(fixGlobalAccessNode).toHaveFocus();
    expect(await screen.findByText(/^Guide:\s*10\.2\.1$/i)).toBeInTheDocument();
  });

  it("closes on Escape and backdrop click but not on dialog clicks", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<CudaTreasureMapModal open={true} onClose={onClose} />);

    const dialog = screen.getByRole("dialog", { name: /CUDA operator treasure route/i });
    const backdrop = dialog.parentElement;

    expect(backdrop).not.toBeNull();

    await user.click(dialog);
    expect(onClose).not.toHaveBeenCalled();

    await user.click(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
