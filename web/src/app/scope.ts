import type { Profile } from "../api";
import { PLATFORM_LABELS, SITE_LABELS } from "../lib/labels";

/** 和后端 profile_filters 的规则一致：只主营一个平台 / 站点时，没说明的问题才自动按它检索 */
export function scopeOf(profile: Profile | null): string | null {
  if (!profile) return null;
  const parts: string[] = [];
  if (profile.platforms.length === 1) parts.push(PLATFORM_LABELS[profile.platforms[0]] ?? profile.platforms[0]);
  if (profile.sites.length === 1) parts.push(`${SITE_LABELS[profile.sites[0]] ?? profile.sites[0]}站`);
  return parts.length ? parts.join(" · ") : null;
}
