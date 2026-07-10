# MLflow（未实现占位）

本目录为 **规划占位**，当前流水线**未**集成 MLflow。训练/推理指标以 `runs/<run_name>/` 下的 `metrics.json`、`best_threshold.json` 与 `outputs/tables/*_deployment_kpi.csv` 为准。

若后续接入，建议最小跟踪字段：

- params：实验配置、模型超参
- metrics：val/test f1、precision、recall、roc_auc
- artifacts：PR 曲线、混淆矩阵、checkpoint 路径
