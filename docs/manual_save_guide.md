# 手动保存 Amazon 页面指南

Amazon 的使用条件只允许个人、非商业使用，并禁止使用机器人和数据挖掘工具，所以这 40 个页面需要你在自己的浏览器里手动保存。预计用时 30–40 分钟，可以分几次完成。

## 保存步骤（Chrome / Edge 相同）

1. 点击下表里的链接，在浏览器中打开页面，不需要登录。
2. 按 `Ctrl + S`。
3. **保存类型**选 **网页，仅 HTML**（不要选「网页，全部」，那样会多出一个素材文件夹）。
4. **文件名**填表格里的编号，例如 `m01`，浏览器会自动补上 `.html`。
5. **保存位置**选 `D:\my_project\data\raw\manual\`（目录已经建好）。

保存完一批后，在项目根目录运行下面的命令，把它们登记进抓取记录。可以随时反复运行，已经登记过的会跳过：

```bash
uv run python scripts/import_manual.py
```

如果提示「正文只有 N 字」，说明保存的页面不完整。等页面完全加载后，重新按 `Ctrl + S` 保存一次。

## 页面清单

| 编号 | 站点 | 页面 |
|---|---|---|
| `m01` | 通用 | [跨境电商政策_全球开店开店规则_侵犯知识产权](https://gs.amazon.cn/policy) |
| `m02` | 通用 | [亚马逊全球开店_亚马逊开店规则_对卖家权限被删除提出申诉](https://gs.amazon.cn/policy/appeal-the-removal-of-selling-privileges) |
| `m03` | 通用 | [亚马逊全球开店_亚马逊开店规则_侵犯知识产权](https://gs.amazon.cn/policy/intellectual-property-violations) |
| `m04` | 通用 | [亚马逊全球开店_亚马逊开店规则_确保商品质量](https://gs.amazon.cn/policy/product-quality) |
| `m05` | 通用 | [亚马逊开店_产品合规政策_确定哪些商品不能销售](https://gs.amazon.cn/policy/product-safety) |
| `m06` | 通用 | [亚马逊全球开店_亚马逊开店规则_禁止的卖家行为](https://gs.amazon.cn/policy/prohibited-actions) |
| `m07` | 通用 | [亚马逊跨境电商开店流程及费用_注册需要什么资料条件](https://gs.amazon.cn/sell) |
| `m08` | 通用 | [亚马逊怎么开店注册_开店前需要准备什么资料_如何通过审核](https://gs.amazon.cn/sell/cny) |
| `m09` | 美国 | [如何注册成为亚马逊卖家](https://sell.amazon.com/sell/registration-guide) |
| `m10` | 美国 | [在亚马逊商城开店需要支付多少费用？](https://sell.amazon.com/pricing) |
| `m11` | 美国 | [亚马逊美国FBA常见问题_北美FBA收费标准](https://gs.amazon.cn/north-america/fba/faq) |
| `m12` | 美国 | [重要通知 | 2020年亚马逊美国站费用变更汇总|政策新规](https://gs.amazon.cn/north-america/fba/pricing) |
| `m13` | 美国 | [亚马逊物流](https://sell.amazon.com/fulfillment-by-amazon) |
| `m14` | 美国 | [亚马逊物流国际](https://sell.amazon.com/fulfillment-by-amazon/fba-international) |
| `m15` | 美国 | [亚马逊物流远程配送](https://sell.amazon.com/fulfillment-by-amazon/remote-fulfillment) |
| `m16` | 美国 | [亚马逊物流多渠道配送 (MCF)](https://sell.amazon.com/fulfillment-by-amazon/fba-multichannel) |
| `m17` | 美国 | [亚马逊 FBM（卖家自配送）](https://sell.amazon.com/programs/fulfilled-by-merchant) |
| `m18` | 通用 | [MFN_亚马逊卖家自配送_跨境自配送](https://gs.amazon.cn/mfn) |
| `m19` | 通用 | [“购买配送”服务（Buy Shipping）_亚马逊卖家自配送_跨境自配送](https://gs.amazon.cn/mfn/buyshipping) |
| `m20` | 美国 | [亚马逊全球物流](https://sell.amazon.com/programs/amazon-global-logistics) |
| `m21` | 美国 | [跨境承运伙伴方案 (SEND)](https://sell.amazon.com/programs/seller-export-and-delivery) |
| `m22` | 美国 | [亚马逊智能卫星仓 (AWD)](https://sell.amazon.com/programs/warehousing) |
| `m23` | 通用 | [亚马逊物流库存管理_冗余库存清销四步法](https://gs.amazon.cn/logistics/inventory/destock/overview) |
| `m24` | 通用 | [FBA入库分仓决策工具包](https://gs.amazon.cn/logistics/inventory/ipsf-desicion) |
| `m25` | 通用 | [补货难题自查诊断器](https://gs.amazon.cn/fba/inbound-clinic) |
| `m26` | 通用 | [亚马逊退货管理工具包](https://gs.amazon.cn/c-return) |
| `m27` | 美国 | [亚马逊品牌注册](https://sell.amazon.com/brand-registry) |
| `m28` | 美国 | [亚马逊 Transparency](https://sell.amazon.com/brand-registry/transparency) |
| `m29` | 美国 | [Project Zero](https://sell.amazon.com/brand-registry/project-zero) |
| `m30` | 美国 | [知识产权加速器](https://sell.amazon.com/programs/ip-accelerator) |
| `m31` | 中东 | [VAT中东_阿联酋VAT_沙特增值税官方解读](https://gs.amazon.cn/middle-east/uae-vat) |
| `m32` | 美国 | [亚马逊美国站_加拿站开店官网_北美卖家登录入口](https://gs.amazon.cn/north-america) |
| `m33` | 中东 | [亚马逊中东站_沙特跨境电商注册_阿联酋站怎么样_开网店需要哪些流程](https://gs.amazon.cn/middle-east) |
| `m34` | 欧洲 | [扩展至欧洲](https://sell.amazon.com/global-selling/europe) |
| `m35` | 日本 | [扩展至日本](https://sell.amazon.com/global-selling/japan) |
| `m36` | 美国 | [如何从其他国家/地区在亚马逊美国商城销售商品](https://sell.amazon.com/global-selling/international-to-usa) |
| `m37` | 通用 | [亚马逊的广告关键词怎么设置优化,怎么优化关键词排名](https://gs.amazon.cn/sponsored-products/keywordsoptimizationseocontent) |
| `m38` | 美国 | [通过 Amazon Vine 获得买家评论](https://sell.amazon.com/programs/vine) |
| `m39` | 美国 | [A+ 商品描述](https://sell.amazon.com/tools/a-content) |
| `m40` | 美国 | [Customer Service by Amazon (CSBA)](https://sell.amazon.com/programs/customer-service-by-amazon) |
