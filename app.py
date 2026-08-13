import os
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from redact import run_pipeline

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for flash messages

ALLOWED_EXTENSIONS = {'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/redact', methods=['POST'])
def redact_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create a temporary directory to process the files securely
        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / filename
        
        # Name the output file with "_redacted" suffix
        stem = input_path.stem
        ext = input_path.suffix
        output_filename = f"{stem}_redacted{ext}"
        output_path = Path(temp_dir) / output_filename
        detections_path = Path(temp_dir) / "detections.json"
        
        # Save the uploaded file
        file.save(input_path)
        
        try:
            # Run our redaction pipeline
            from redact import ALL_CATEGORIES
            run_pipeline(
                input_path=input_path, 
                output_path=output_path, 
                seed=42, 
                dry_run=False, 
                categories=ALL_CATEGORIES, 
                detections_path=detections_path
            )
            
            # Send the redacted file back to the user
            return send_file(
                output_path, 
                as_attachment=True,
                download_name=output_filename,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        except Exception as e:
            flash(f"Error processing document: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Only .docx is allowed.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Used only for local development
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
