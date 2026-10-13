# 图文 DSpark 监督所有已有 assistant 轮次

用户要求图文 DSpark 学习所有已有 assistant 回答，包括 reasoning_content 和 tool_calls，因此选择保留历史推理内容并显式监督各轮，而不沿用可能因 thinking 内容仅监督最后一轮的自动规则。user/tool 消息与图片 token 保留为上下文，不作为回答预测目标；prefill 后生成的单 token 不参与监督。这会增加完整会话的处理成本，但避免丢弃用户指定的多轮训练目标；该选择不改变现有纯文本示例的默认行为。
