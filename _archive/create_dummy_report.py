from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_dummy_report(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Diagnostic Lab Report: Kidney Function Test")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 140, "Patient Name: John Doe")
    c.drawString(100, height - 160, "Age: 52 years")
    c.drawString(300, height - 160, "Gender: Male")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 200, "Biometric Results:")
    
    c.setFont("Helvetica", 11)
    results = [
        ("Serum Creatinine", "1.8 mg/dL", "(High)"),
        ("Blood Pressure", "145/95 mmHg", "(Hypertension)"),
        ("Albumin (al)", "2+", "(Abnormal)"),
        ("Sugar (su)", "0", "(Normal)"),
        ("Hemoglobin", "11.5 g/dL", "(Low)"),
        ("Sodium (sod)", "132 mEq/L", "(Low)"),
        ("Potassium (pot)", "4.8 mEq/L", "(Normal)"),
        ("Specific Gravity (sg)", "1.010", ""),
    ]
    
    y = height - 230
    for label, val, note in results:
        c.drawString(120, y, f"{label}: {val} {note}")
        y -= 20
        
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, 100, "Note: This is a synthetic report for AI verification purposes.")
    
    c.showPage()
    c.save()
    print(f"Dummy report created: {filename}")

if __name__ == "__main__":
    create_dummy_report("sample_kidney_report.pdf")
