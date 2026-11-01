import { useEffect, useRef, useState } from "react";

/**
 * 把一阵一阵到达的流式文本匀速放出来。
 * 按经过的时间算放多少字，不按帧数：浏览器降帧（后台标签页、省电模式）时每帧多放一些，不会越积越多。
 * - 积压按指数衰减追赶：每 16ms 放出积压的 1/10，积压越多放得越快；
 * - 至少每秒 40 个字，积压很少时也是连续打字的感觉。
 * 流式结束后会把剩下的放完，不会突然跳到全文。animate 为 false（历史消息）时直接显示全文。
 */
const FRAME_MS = 1000 / 60;
const CATCH_UP = 0.1;
const MIN_CHARS_PER_MS = 0.04;

export function useSmoothText(target: string, animate: boolean): { text: string; caughtUp: boolean } {
  const [shown, setShown] = useState(animate ? "" : target);
  const shownRef = useRef(shown);
  const targetRef = useRef(target);
  targetRef.current = target;

  useEffect(() => {
    if (!animate) return;
    let frame = 0;
    let last = performance.now();
    let carry = 0; // 不足一个字的部分攒到下一帧
    const tick = (now: number) => {
      const elapsed = Math.min(now - last, 1000);
      last = now;
      const goal = targetRef.current;
      let current = shownRef.current;
      // 最终版回答会删掉越界的 [n]，和流式拼出来的不完全一样：退到公共前缀再接着放
      if (!goal.startsWith(current)) {
        let i = 0;
        while (i < current.length && i < goal.length && current[i] === goal[i]) i++;
        current = goal.slice(0, i);
      }
      const backlog = goal.length - current.length;
      if (backlog > 0) {
        const decay = 1 - Math.pow(1 - CATCH_UP, elapsed / FRAME_MS);
        carry += Math.max(backlog * decay, elapsed * MIN_CHARS_PER_MS);
        const step = Math.min(backlog, Math.floor(carry));
        carry -= step;
        current = goal.slice(0, current.length + step);
      } else {
        carry = 0;
      }
      if (current !== shownRef.current) {
        shownRef.current = current;
        setShown(current);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [animate]);

  const text = animate ? shown : target;
  return { text, caughtUp: text.length >= target.length };
}
