"""Configuration management for the linkages pipeline.

Loads API keys from .env and provides sensible path defaults.
All other modules import Settings from here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # API keys (from .env)
    openai_api_key: str = field(
        default_factory=lambda: os.environ["OPENAI_API_KEY"]
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ["ANTHROPIC_API_KEY"]
    )

    # Project root: defaults to 2 levels up from this file (modules/config.py)
    # Override with LINKAGES_PROJECT_ROOT env var for SLURM or non-standard layouts
    project_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "LINKAGES_PROJECT_ROOT",
                str(Path(__file__).resolve().parent.parent),
            )
        )
    )

    # Model configuration
    embedding_model_name: str = "dell-research-harvard/lt-un-data-fine-fine-en"
    claude_model: str = "claude-3-5-haiku-latest"
    gpt_model: str = "gpt-4o-mini"

    # Pipeline parameters
    hs_sheet: str = "HS12"
    hs_level: str = "4"
    top_k_total: int = 25
    top_k_bert: int = 5
    checkpoint_every: int = 10

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"

    @property
    def hs_table_path(self) -> Path:
        return self.raw_dir / "HSCodeandDescription.xlsx"

    @property
    def embeddings_path(self) -> Path:
        return self.intermediate_dir / "hs12_4_embeddings.npy"

    @property
    def base_df_path(self) -> Path:
        return self.intermediate_dir / "base_df.parquet"
