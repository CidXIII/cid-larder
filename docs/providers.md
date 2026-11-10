# Model providers

Cid-Larder talks to any endpoint that speaks the OpenAI API format.
Two endpoints are configured separately: one for chat, one for embeddings.
They can be the same provider or different ones.

## Examples

Model names change often. Treat these as shapes, not as the current truth.

| Provider | base_url | 说明 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | 只有对话模型，没有 embedding |
| Zhipu GLM | `https://open.bigmodel.cn/api/paas/v4` | 对话 + embedding |
| OpenAI | `https://api.openai.com/v1` | 对话 + embedding |
| SiliconFlow | `https://api.siliconflow.cn/v1` | 对话 + embedding |
| Ollama | `http://127.0.0.1:11434/v1` | 本地，取决于你 pull 了什么 |

**不在这里写模型名。** 模型 ID 几个月就换一批，写死在文档里等于写错。候选清单在
`app/core/providers.json`，界面上的「拉取模型」会直接问服务商 `GET /models`，
用你的 key 拿当前真正可用的列表。

## Notes

- **Chat and embeddings may come from different vendors.** DeepSeek serves
  chat but has no embeddings endpoint, so a second provider is required for
  vectors. Leave the `EMBEDDING_*` block empty only if your chat provider
  also serves embeddings.
- **Switching the embedding model invalidates stored vectors.** Different
  models produce different dimensions and different spaces. The app records
  the model name and dimension, and asks you to re-embed when they change.
- **Keys stay local.** Keys typed into the page are written to
  `data/config.json` in plain text (the folder is git-ignored); `.env` keys
  take precedence and are never written back. Neither file is committed, and
  keys are never logged in full.
