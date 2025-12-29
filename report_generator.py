from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io
import datetime
import re

def clean_markdown(text):
    """
    Converts simple Markdown to ReportLab XML tags or clean text.
    Handles bold (**text**) and headers.
    """
    if not text: return ""
    
    # Text replacement for bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Remove # headers styling (ReportLab handles headers via styles)
    text = re.sub(r'#+\s*', '', text)
    
    # Replace newlines with <br/> for HTML-like flow in Paragraphs if needed, 
    # but usually we split by newline and make separate paragraphs.
    return text

def create_pdf_report(patient_name, prediction_data, prescription_text, vitals_dict=None):
    """
    Generates a PDF report buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    Story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=20,
        textColor=colors.darkblue
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkslategray,
        borderPadding=5,
        borderColor=colors.lightgrey,
        borderWidth=0,
        backColor=colors.whitesmoke
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    warning_style = ParagraphStyle(
        'WarningStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.red,
        alignment=TA_CENTER
    )

    # --- CONTENT BUILDING ---
    
    # 1. Title & Header
    Story.append(Paragraph("KidneyPred AI Medical Report", title_style))
    Story.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
    Story.append(Spacer(1, 10))
    
    # 2. Patient Info Table
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    p_info = [
        ["Patient Name:", patient_name],
        ["Date:", date_str],
        ["Report ID:", f"KP-{datetime.datetime.now().strftime('%f')[:6]}"]
    ]
    t = Table(p_info, colWidths=[100, 300])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    Story.append(t)
    Story.append(Spacer(1, 20))
    
    # 3. Prediction Summary
    Story.append(Paragraph("Diagnostic Summary", header_style))
    
    consensus = prediction_data.get('consensus', 'Analysis')
    confidence = prediction_data.get('confidence', 0)
    
    status_color = "red" if "CKD" in consensus else "green"
    status_text = f"<b>Status: <font color='{status_color}'>{consensus}</font></b> (Confidence: {confidence:.1%})"
    Story.append(Paragraph(status_text, styles['Normal']))
    
    if vitals_dict:
        Story.append(Spacer(1, 10))
        Story.append(Paragraph("<b>Key Vitals:</b>", styles['Normal']))
        # Create a grid for vitals
        v_data = []
        row = []
        for k, v in vitals_dict.items():
            if v != 0 and v != '0' and v != 0.0:
                row.append(f"{k}: {v}")
            if len(row) == 3:
                v_data.append(row)
                row = []
        if row: v_data.append(row)
        
        if v_data:
            vt = Table(v_data, colWidths=[150, 150, 150])
            vt.setStyle(TableStyle([
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.dimgrey),
            ]))
            Story.append(vt)

    # 4. Clinical Prescription
    Story.append(Paragraph("Clinical Assessment & Plan", header_style))
    
    # Process text line by line to handle formatting decently
    lines = prescription_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            Story.append(Spacer(1, 6))
            continue
            
        # Handle Headers mapped to Bold Paragraphs
        if line.startswith('##') or line.startswith('###'):
            clean_line = line.replace('#', '').strip()
            Story.append(Paragraph(f"<b>{clean_line}</b>", styles['Heading3']))
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point
            clean_line = clean_markdown(line[2:])
            Story.append(Paragraph(f"• {clean_line}", ParagraphStyle('Bullet', parent=normal_style, leftIndent=20)))
        else:
            # Normal text
            clean_line = clean_markdown(line)
            Story.append(Paragraph(clean_line, normal_style))
            
    # 5. Judges Score
    if 'score' in prediction_data:
        Story.append(Spacer(1, 20))
        Story.append(Paragraph("Quality Assurance", header_style))
        Story.append(Paragraph(f"AI Quality Score: <b>{prediction_data['score']}/10</b>", normal_style))

    # 6. Disclaimer Footer
    Story.append(Spacer(1, 40))
    Story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disclaimer = ("DISCLAIMER: This report is generated by an Artificial Intelligence system (KidneyPred AI). "
                  "It is intended for educational and informational purposes only and does NOT constitute medical advice, "
                  "diagnosis, or treatment. Always consult a qualified healthcare provider.")
    Story.append(Paragraph(disclaimer, warning_style))
    
    doc.build(Story)
    buffer.seek(0)
    return buffer

def create_chat_log_pdf(patient_name, messages):
    """
    Generates a PDF buffer for the chat transcript.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    Story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=20, spaceAfter=20, textColor=colors.darkblue
    )
    Story.append(Paragraph("KidneyPred AI - Consultation Transcript", title_style))
    Story.append(HRFlowable(width="100%", thickness=1, color=colors.darkblue))
    Story.append(Spacer(1, 10))
    
    # Meta
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    p_info = f"<b>Patient:</b> {patient_name} <br/> <b>Date:</b> {date_str}"
    Story.append(Paragraph(p_info, styles['Normal']))
    Story.append(Spacer(1, 20))
    
    # Messages
    user_style = ParagraphStyle('User', parent=styles['Normal'], backColor=colors.whitesmoke, borderPadding=5, spaceAfter=10)
    ai_style = ParagraphStyle('AI', parent=styles['Normal'], backColor=colors.aliceblue, borderPadding=5, spaceAfter=10)
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Format role header
        role_text = "PATIENT" if role == "user" else "NEPHRO AI"
        role_style = user_style if role == "user" else ai_style
        
        Story.append(Paragraph(f"<b>{role_text}</b>:", role_style))
        
        # Process content line by line for proper formatting
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                Story.append(Spacer(1, 4))
                continue
            
            # Handle headers (## or ###)
            if line.startswith('##') or line.startswith('###'):
                clean_line = line.replace('#', '').strip()
                # Remove emoji characters that don't render well in PDF
                clean_line = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_line)
                Story.append(Spacer(1, 8))
                Story.append(Paragraph(f"<b>{clean_line}</b>", styles['Heading4']))
            
            # Handle bullet points
            elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                clean_line = clean_markdown(line[2:])
                clean_line = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_line)
                bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], leftIndent=20, fontSize=9, leading=12)
                Story.append(Paragraph(f"• {clean_line}", bullet_style))
            
            # Handle numbered lists (1. 2. etc)
            elif re.match(r'^\d+\.', line):
                clean_line = clean_markdown(line)
                clean_line = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_line)
                num_style = ParagraphStyle('Numbered', parent=styles['Normal'], leftIndent=20, fontSize=9, leading=12)
                Story.append(Paragraph(clean_line, num_style))
            
            # Handle separator lines
            elif line.startswith('---'):
                Story.append(HRFlowable(width="80%", thickness=0.5, color=colors.lightgrey))
                Story.append(Spacer(1, 6))
            
            # Normal text
            else:
                clean_line = clean_markdown(line)
                # Remove emoji characters that cause black squares
                clean_line = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_line)
                # Also remove other problematic Unicode
                clean_line = re.sub(r'[■▪◾]', '-', clean_line)
                content_style = ParagraphStyle('Content', parent=styles['Normal'], fontSize=9, leading=12)
                Story.append(Paragraph(clean_line, content_style))
        
        Story.append(Spacer(1, 10))

    doc.build(Story)
    buffer.seek(0)
    return buffer
