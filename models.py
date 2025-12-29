# Pydantic Models for Input Validation
# Provides type safety and validation for patient data

from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PatientData(BaseModel):
    """Validated patient data model for CKD risk prediction."""
    
    age: int = Field(ge=0, le=120, default=0, description="Patient age in years")
    bp: float = Field(ge=0, le=300, default=0, description="Blood pressure (diastolic, mmHg)")
    sg: float = Field(ge=1.0, le=1.05, default=1.015, description="Specific gravity of urine")
    al: float = Field(ge=0, le=5, default=0, description="Albumin level (0-5 scale)")
    su: float = Field(ge=0, le=5, default=0, description="Sugar level (0-5 scale)")
    rbc: str = Field(default="normal", description="Red blood cells (normal/abnormal)")
    pc: str = Field(default="normal", description="Pus cells (normal/abnormal)")
    pcc: str = Field(default="notpresent", description="Pus cell clumps (present/notpresent)")
    ba: str = Field(default="notpresent", description="Bacteria (present/notpresent)")
    bgr: float = Field(ge=0, le=500, default=100, description="Blood glucose random (mg/dL)")
    bu: float = Field(ge=0, le=300, default=30, description="Blood urea (mg/dL)")
    sc: float = Field(ge=0, le=20, default=1.0, description="Serum creatinine (mg/dL)")
    sod: float = Field(ge=100, le=180, default=140, description="Sodium (mEq/L)")
    pot: float = Field(ge=2, le=10, default=4.5, description="Potassium (mEq/L)")
    hemo: float = Field(ge=3, le=20, default=13, description="Hemoglobin (g/dL)")
    pcv: float = Field(ge=10, le=60, default=40, description="Packed cell volume (%)")
    wc: float = Field(ge=2000, le=30000, default=8000, description="White blood cell count")
    rc: float = Field(ge=2, le=8, default=5, description="Red blood cell count (millions/cmm)")
    htn: str = Field(default="no", description="Hypertension (yes/no)")
    dm: str = Field(default="no", description="Diabetes mellitus (yes/no)")
    cad: str = Field(default="no", description="Coronary artery disease (yes/no)")
    appet: str = Field(default="good", description="Appetite (good/poor)")
    pe: str = Field(default="no", description="Pedal edema (yes/no)")
    ane: str = Field(default="no", description="Anemia (yes/no)")
    grf: float = Field(ge=0, le=150, default=90, description="Glomerular filtration rate (mL/min)")
    
    @field_validator('rbc', 'pc', mode='before')
    @classmethod
    def normalize_rbc_pc(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
        return 'normal' if v in ['normal', 'n', ''] else 'abnormal'
    
    @field_validator('pcc', 'ba', mode='before')
    @classmethod
    def normalize_present(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
        return 'present' if v in ['present', 'p', 'yes'] else 'notpresent'
    
    @field_validator('htn', 'dm', 'cad', 'pe', 'ane', mode='before')
    @classmethod
    def normalize_yesno(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
        return 'yes' if v in ['yes', 'y', '1', 'true'] else 'no'
    
    @field_validator('appet', mode='before')
    @classmethod
    def normalize_appet(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
        return 'good' if v in ['good', 'g', ''] else 'poor'
    
    def to_dict(self):
        """Convert to dictionary for session state."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary with validation."""
        return cls(**data)
    
    def get_ckd_stage(self) -> str:
        """Estimate CKD stage based on GFR."""
        gfr = self.grf
        if gfr >= 90:
            return "Stage 1 (Normal or high GFR)"
        elif gfr >= 60:
            return "Stage 2 (Mild decrease)"
        elif gfr >= 45:
            return "Stage 3a (Mild to moderate decrease)"
        elif gfr >= 30:
            return "Stage 3b (Moderate to severe decrease)"
        elif gfr >= 15:
            return "Stage 4 (Severe decrease)"
        else:
            return "Stage 5 (Kidney failure)"


class ChatExportData(BaseModel):
    """Model for chat export functionality."""
    
    patient_name: Optional[str] = Field(default="Patient", description="Patient name for report")
    consultation_date: Optional[str] = Field(default=None, description="Date of consultation")
    messages: list = Field(default_factory=list, description="Chat messages")
    model_predictions: Optional[dict] = Field(default=None, description="ML model predictions")
    council_report: Optional[str] = Field(default=None, description="Council synthesis report")


if __name__ == "__main__":
    # Test validation
    try:
        patient = PatientData(age=65, sc=2.5, grf=28, htn="yes", dm="yes")
        print(f"Valid patient: {patient}")
        print(f"CKD Stage: {patient.get_ckd_stage()}")
    except Exception as e:
        print(f"Validation error: {e}")
