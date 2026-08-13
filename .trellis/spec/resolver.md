# Resolver

Weadge 唯一活跃 production 主线（research v1 FROZEN）。

- 策略唯一权威文档: `docs/resolver.md`（与代码冲突时以文档为准）
- 代码在 `src/weadge/resolver/`，业务逻辑绝不 import `research/`
- 复用 `domain/` primitives（time 的 as-of 不变式是硬规则）
- V0 只做冷端 LOCKED：`observed_max >= cap_high + locked_buffer_c`
- 禁止：概率模型（V1）、forecast（V2）、trade mode（v1+）、多城市（巴黎首发）
- 纯函数核心（evaluate/find_edges）必须离线可测，fixture 用真实 API 样本
- 新增城市 = 加 config + 确认结算站（PM 规则里 resolutionSource 为准）
