import type { SimtOperatorDiff, SimtOperatorVersionIndex } from "../types";

export async function fetchSimtOperatorVersions(
  operator: string,
  signal?: AbortSignal
): Promise<SimtOperatorVersionIndex> {
  const params = new URLSearchParams({ operator });
  const response = await fetch(`/api/simt-versions?${params.toString()}`, { signal });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `SIMT version request failed with status ${response.status}`);
  }
  return (await response.json()) as SimtOperatorVersionIndex;
}

export async function fetchSimtOperatorDiff(
  operator: string,
  baseVersion: string,
  compareVersion: string,
  signal?: AbortSignal
): Promise<SimtOperatorDiff> {
  const params = new URLSearchParams({
    operator,
    base_version: baseVersion,
    compare_version: compareVersion
  });
  const response = await fetch(`/api/simt-diff?${params.toString()}`, { signal });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `SIMT diff request failed with status ${response.status}`);
  }
  return (await response.json()) as SimtOperatorDiff;
}
