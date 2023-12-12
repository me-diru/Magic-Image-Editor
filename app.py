from flask import Flask, render_template, request, send_file
import numpy as np
# Import your seam carving logic here
from seam_carving_methods import *
import cv2
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if request.method == 'POST':
        image_file = request.files['image']
        method = request.form['method']  # forward, backward, or other methods
        num_rows = int(request.form.get('numRows', 0))  # Number of rows to remove
        num_columns = int(request.form.get('numColumns', 0))  # Number of columns to remove
    
        processed_image = ""
        image = ""
      
        
        if image_file:  # Check if the file exists
        # Read the file's contents into memory
            filestr = image_file.read()

            # Convert string data to numpy array
            npimg = np.frombuffer(filestr, np.uint8)

            # Convert numpy array to image
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            original_image_path = f'Images/original/{image_file.filename}'
            original_image_path_1 = f'static/Images/original/{image_file.filename}'

            cv2.imwrite(original_image_path_1, image) 

            if method == "forward_vertical":
                processed_image = remove_vertical_seams(image, num_columns, compute_forward_energy_vertical)
            elif method == "forward_horizontal":
                processed_image = remove_horizontal_seams(image, num_rows, compute_forward_energy_horizontal)
            elif method == "backward_vertical":
                processed_image = remove_vertical_seams(image, num_columns, compute_backward_energy_vertical)
            elif method == "backward_horizontal":
                processed_image = remove_horizontal_seams(image, num_rows, compute_backward_energy_horizontal)
            elif method == "greedy":
                processed_image = adaptive_seam_removal(image, num_columns, num_rows)
            elif method == "monte_carlo":
                best_sequence, _ = monte_carlo_seam_carving(image, num_columns, num_rows, iterations=10)
                processed_image = get_image_from_sequence(best_sequence, image)
                


            # print(method, image_file.filename, )
        
        processed_image_path_1 = 'static/Images/processed/'+method+"_"+image_file.filename
        processed_image_path = 'Images/processed/'+method+"_"+image_file.filename
        
        cv2.imwrite(processed_image_path_1, processed_image)
        # processed_image.save(processed_image_path)

        return render_template('results.html', original_image=original_image_path, processed_image=processed_image_path)

if __name__ == '__main__':
    app.run(debug=True)
