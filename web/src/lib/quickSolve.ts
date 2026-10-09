/** The "Quick test solve (skip exports)" box on the layer review screen; every solve button follows it. */
const KEY = "beeline.quickSolve";

export function quickSolveOn(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function setQuickSolve(v: boolean): void {
  try {
    localStorage.setItem(KEY, v ? "1" : "0");
  } catch {
    /* per-browser convenience only */
  }
}
