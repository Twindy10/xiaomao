// 模拟接口模式下，地址带 #demo 时直接以演示账号登录，方便截图检查（只影响 npm run dev:mock）。
// 模拟接口在加载时就读了登录状态（打包后它在公共代码块里，比这里先执行），所以写完标记要重新加载一次
if (import.meta.env.VITE_MOCK === "1" && location.hash.startsWith("#demo")) {
  const fresh = localStorage.getItem("xm.mock.session") !== "1";
  localStorage.setItem("xm.mock.session", "1");
  history.replaceState(null, "", location.pathname + location.search + location.hash.replace(/^#demo/, "#"));
  if (fresh) location.reload();
}
