from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from GradCam import inference
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # 2MB Limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename to avoid overwriting
            unique_id = uuid.uuid4().hex
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{unique_id}.{ext}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            img = Image.open(save_path).convert('RGB')

            # Generate prediction
            output, confidence = inference.inference(image=img)
            
            # Generate overlay heatmap
            # overlay_img = overlay.Overlay_heatmap(img)
            # overlay_hm_img = Image.fromarray(overlay_img, 'RGB')
            
            # Save overlay image
            # overlay_filename = f"overlay_{filename}"
            # overlay_save_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            # overlay_hm_img.save(overlay_save_path)

            return render_template('result.html', 
                                original_image=url_for('static', filename=f"uploads/{filename}"),
                                # overlay_image=url_for('static', filename=f"uploads/{overlay_filename}"),
                                prediction=output,
                                confidence=f"{confidence:.2f}%")
        
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port= 5000)
