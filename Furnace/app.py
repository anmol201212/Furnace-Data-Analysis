from flask import Flask, render_template, request, redirect, url_for
import os
from model_training import read_csv_with_statistics
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

@app.route('/')
def index():
    csv_data, statistics = read_csv_with_statistics(r'data/FurnaceData.csv')
    screen2_data = "Data for Screen 2"
    screen3_data = "Data for Screen 3"

    # labels = [row for row in statistics]
    # values = [statistics[key] for key in statistics]

    chart_data = {
        'headers': ['PreHeatZone', 'HeatZone1', 'HeatZone2', 'SockingZone'],
        'labels': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'values': [
            [3271, 3271, 3271, 3271],
            [440.82084989, 610.54631611, 689.62335677, 1035.71996331],
            [85.77075697, 54.07767756, 31.75498514, 60.07068466],
            [1., 550., 0., 970.],
            [415., 585., 665., 1010.],
            [439., 610., 690., 1034.],
            [465., 635., 715., 1060.],
            [5000., 3200., 750., 4000.]
        ]
    }

    return render_template('index.html', csv_data=csv_data, statistics=statistics,chart_data=chart_data, screen2_data=screen2_data, screen3_data=screen3_data)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return 'File uploaded successfully!'
    else:
        return 'Error uploading file.'



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', debug=True)
