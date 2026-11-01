import { mockApi } from "./mock";
import { realApi } from "./real";
import type { Api } from "./types";

export const api: Api = import.meta.env.VITE_MOCK === "1" ? mockApi : realApi;
export { sessionEvents } from "./real";
export * from "./types";
