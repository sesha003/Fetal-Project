"""fecg — fetal ECG/FHR extraction from abdominal ECG, focused on the maternal/
fetal QRS overlap case."""
from .config import PipelineConfig
from .io import (Record, synthesize_record, synthetic_cohort,
                 load_adfecgdb_record, load_physionet_record,
                 load_edf_record, read_edf_signals)
from .pipeline import run_pipeline, leave_one_record_out
from .suppression import (Suppressor, TemplateSubtraction, ICASuppressor,
                          AdaptiveSuppressor, KalmanTemplateSuppressor)

__version__ = "0.1.0"
__all__ = [
    "PipelineConfig", "Record", "synthesize_record", "synthetic_cohort",
    "load_adfecgdb_record", "load_physionet_record",
    "load_edf_record", "read_edf_signals",
    "run_pipeline", "leave_one_record_out",
    "Suppressor", "TemplateSubtraction", "ICASuppressor",
    "AdaptiveSuppressor", "KalmanTemplateSuppressor",
]
