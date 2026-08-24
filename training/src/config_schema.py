import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ModelConfig(BaseModel):
    base_model_name: str = Field(
        default_factory=lambda: os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    torch_dtype: str = Field(default="bfloat16")
    max_seq_length: int = Field(default=1024, ge=128, le=32768)
    trust_remote_code: bool = Field(default=True)

    @field_validator("base_model_name", mode="before")
    @classmethod
    def populate_from_env_if_none(cls, v: str | None) -> str:
        if not v:
            return os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        return v


class DatasetConfig(BaseModel):
    raw_data_path: str = Field(default="training/data/raw/sample_financial_alpaca.jsonl")
    processed_dir: str = Field(default="training/data/processed")
    train_split_ratio: float = Field(default=0.90, gt=0.0, lt=1.0)
    val_split_ratio: float = Field(default=0.10, gt=0.0, lt=1.0)
    calibration_samples: int = Field(default=128, ge=16)
    format: str = Field(default="alpaca")

    @field_validator("val_split_ratio")
    @classmethod
    def validate_splits(cls, v: float, info: ValidationInfo) -> float:
        train_ratio = info.data.get("train_split_ratio", 0.90)
        if round(train_ratio + v, 4) != 1.0:
            raise ValueError(f"train_split_ratio ({train_ratio}) + val_split_ratio ({v}) must sum to 1.0")
        return v


class QLoRAConfig(BaseModel):
    r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    bias: str = Field(default="none")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    task_type: str = Field(default="CAUSAL_LM")
    quant_type: str = Field(default="nf4")
    double_quant: bool = Field(default=True)


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="artifacts/runs/dev")
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2.0e-4, gt=0.0)
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    num_train_epochs: int = Field(default=1, ge=1)
    max_steps: int = Field(default=-1)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=25, ge=1)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)
    optim: str = Field(default="paged_adamw_8bit")


class EvaluationConfig(BaseModel):
    batch_size: int = Field(default=4, ge=1)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "json_validity_rate",
            "schema_compliance_rate",
            "field_level_accuracy"
        ]
    )


class IntentRoutingConfig(BaseModel):
    adapter_name: str = Field(default="financial_adapter")
    description: str = Field(default="Fine-Tuned LoRA Adapter for Vietnamese Financial & Transaction Extraction")
    keywords: list[str] = Field(
        default_factory=lambda: [
            "chuyển tiền", "chuyển khoản", "stk", "số tài khoản", "tài khoản",
            "giao dịch", "vnd", "nạp tiền", "rút tiền", "sao kê", "thẻ tín dụng"
        ]
    )
    regex_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b\d{1,3}(?:[.,]\d{3})*\s*(?:vnd|đ|dong|nghìn|triệu|k)\b",
            r"(?i)\b(?:stk|tk|số thẻ)[:\s]*\d{8,19}\b"
        ]
    )


class PipelineConfig(BaseModel):
    pipeline_name: str = Field(default="financial_transaction_extraction")
    version: str = Field(default="1.0.0")
    seed: int = Field(default=42)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    intent_routing: IntentRoutingConfig = Field(default_factory=IntentRoutingConfig)


def load_pipeline_config(config_path: str) -> PipelineConfig:
    """Loads and validates a pipeline YAML configuration file against the Pydantic schema."""
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        fallback_path = PROJECT_ROOT / config_path
        if fallback_path.exists():
            path = fallback_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary root.")

    return PipelineConfig(**raw_dict)
