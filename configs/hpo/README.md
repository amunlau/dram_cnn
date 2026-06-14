# configs/hpo/

Optuna 超参搜索空间配置，与 `configs/experiments/` 中的实验 YAML 配合使用。

| 文件 | 用途 |
|------|------|
| `hpo.yaml` | 默认 GridCNN 结构搜索（20 trials） |
| `hpo_verify.yaml` | DDP 验证用单 trial（`scripts/verify_ddp.sh`） |

入口：`bash scripts/optuna.sh [实验yaml] [hpo配置]`，默认 `configs/experiments/exp_optuna_smoke.yaml` + `configs/hpo/hpo.yaml`。
