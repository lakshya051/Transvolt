import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
import io
import base64
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.analysis_results = {}
        
    def load_data(self, file_path):
        """Load CSV data from file path"""
        try:
            self.df = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_data_from_upload(self, file):
        """Load CSV data from uploaded file"""
        try:
            self.df = pd.read_csv(file)
            return True
        except Exception as e:
            print(f"Error loading uploaded data: {e}")
            return False
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            return False
            
        try:
            # Convert timestamp
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], 
                                                format='%d-%m-%Y %H:%M', 
                                                errors='coerce')
            
            # If the above format fails, try other common formats
            if self.df['Timestamp'].isna().all():
                self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], 
                                                    errors='coerce')
            
            # Remove rows with invalid timestamps or values
            self.df = self.df.dropna(subset=['Timestamp', 'Values'])
            
            # Sort by timestamp
            self.df = self.df.sort_values('Timestamp')
            
            # Remove outliers using IQR method
            Q1 = self.df['Values'].quantile(0.25)
            Q3 = self.df['Values'].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df['Values'] < (Q1 - 1.5 * IQR)) | 
                               (self.df['Values'] > (Q3 + 1.5 * IQR)))]
            
            return True
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return False
    
    def calculate_moving_averages(self):
        """Calculate moving averages"""
        self.df['MA_5'] = self.df['Values'].rolling(window=5, min_periods=1).mean()
        self.df['MA_1000'] = self.df['Values'].rolling(window=1000, min_periods=1).mean()
        self.df['MA_5000'] = self.df['Values'].rolling(window=5000, min_periods=1).mean()
    
    def find_peaks_and_lows(self):
        """Find local peaks and lows using scipy"""
        values = self.df['Values'].values
        
        # Find peaks
        peaks, peak_properties = find_peaks(values, 
                                          height=np.percentile(values, 10),
                                          distance=max(1, len(values) // 100))
        
        # Find lows (valleys)
        lows, low_properties = find_peaks(-values, 
                                        height=-np.percentile(values, 90),
                                        distance=max(1, len(values) // 100))
        
        # Create DataFrames
        peak_df = self.df.iloc[peaks][['Timestamp', 'Values']].copy()
        low_df = self.df.iloc[lows][['Timestamp', 'Values']].copy()
        
        return peak_df, low_df
    
    def find_values_below_threshold(self, threshold=20):
        """Find instances where values are below threshold"""
        return self.df[self.df['Values'] < threshold][['Timestamp', 'Values']].copy()
    
    def find_accelerating_downward_slopes(self):
        """Find accelerating downward slopes"""
        self.df['diff'] = self.df['Values'].diff()
        self.df['slope_accel'] = self.df['diff'].diff()
        
        accelerating_down = self.df[(self.df['diff'] < 0) & 
                                   (self.df['slope_accel'] < 0)][['Timestamp', 'Values', 'diff', 'slope_accel']].copy()
        
        return accelerating_down
    
    def get_statistics(self):
        """Calculate basic statistics"""
        if self.df is None:
            return {}
            
        stats = {
            'min': self.df['Values'].min(),
            'max': self.df['Values'].max(),
            'mean': self.df['Values'].mean(),
            'median': self.df['Values'].median(),
            'std': self.df['Values'].std(),
            'count': len(self.df)
        }
        return stats
    
    def create_plots(self):
        """Create all plots and return as base64 encoded strings"""
        plots = {}
        
        # Set style for better looking plots
        plt.style.use('seaborn-v0_8')
        
        # 1. Time series with moving averages
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['Timestamp'], self.df['Values'], label='Values', color='blue', alpha=0.7)
        plt.plot(self.df['Timestamp'], self.df['MA_5'], label='5-value MA', color='red', linewidth=2)
        plt.plot(self.df['Timestamp'], self.df['MA_1000'], label='1000-value MA', color='green', linewidth=2)
        plt.plot(self.df['Timestamp'], self.df['MA_5000'], label='5000-value MA', color='purple', linewidth=2)
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title('Values and Moving Averages')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['timeseries'] = self._plot_to_base64()
        
        # 2. Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Values'], bins=20, kde=True, color='skyblue')
        plt.title("Histogram of Values")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plots['histogram'] = self._plot_to_base64()
        
        # 3. Peak detection plot
        peak_df, low_df = self.find_peaks_and_lows()
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['Timestamp'], self.df['Values'], label='Values', color='blue', alpha=0.7)
        if not peak_df.empty:
            plt.scatter(peak_df['Timestamp'], peak_df['Values'], 
                       color='red', s=50, label='Peaks', zorder=5)
        if not low_df.empty:
            plt.scatter(low_df['Timestamp'], low_df['Values'], 
                       color='green', s=50, label='Lows', zorder=5)
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title('Peak and Low Detection')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['peaks'] = self._plot_to_base64()
        
        # 4. Daily values for 12th day
        df_12 = self.df[self.df['Timestamp'].dt.day == 12].copy()
        if not df_12.empty:
            plt.figure(figsize=(12, 6))
            avg_val = self.df['Values'].mean()
            colors = ['red' if v < avg_val else 'green' for v in df_12['Values']]
            
            plt.bar(range(len(df_12)), df_12['Values'], color=colors)
            plt.xlabel("Data Points on 12th Day")
            plt.ylabel("Values")
            plt.title("Values on the 12th Day of Month (Red=Below Average, Green=Above Average)")
            plt.xticks(range(0, len(df_12), max(1, len(df_12)//10)), 
                      rotation=45)
            plt.tight_layout()
            plots['daily'] = self._plot_to_base64()
        else:
            plots['daily'] = None
            
        return plots
    
    def _plot_to_base64(self):
        """Convert matplotlib plot to base64 string"""
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    
    def perform_complete_analysis(self):
        """Perform complete analysis and return results"""
        if not self.preprocess_data():
            return None
            
        # Calculate moving averages
        self.calculate_moving_averages()
        
        # Get statistics
        stats = self.get_statistics()
        
        # Find peaks and lows
        peak_df, low_df = self.find_peaks_and_lows()
        
        # Find values below threshold
        below_20 = self.find_values_below_threshold(20)
        
        # Find accelerating downward slopes
        accel_down = self.find_accelerating_downward_slopes()
        
        # Create plots
        plots = self.create_plots()
        
        results = {
            'statistics': stats,
            'peaks': peak_df.head(20).to_dict('records'),
            'lows': low_df.head(20).to_dict('records'),
            'below_20': below_20.head(20).to_dict('records'),
            'accelerating_down': accel_down.head(20).to_dict('records'),
            'plots': plots,
            'data_shape': self.df.shape
        }
        
        return results

# Initialize analyzer
analyzer = DataAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.csv'):
        try:
            # Load and analyze data
            if analyzer.load_data_from_upload(file):
                results = analyzer.perform_complete_analysis()
                if results:
                    return jsonify(results)
                else:
                    return jsonify({'error': 'Failed to analyze data'}), 500
            else:
                return jsonify({'error': 'Failed to load CSV file'}), 500
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/analyze_sample')
def analyze_sample():
    """Analyze sample data if available"""
    if os.path.exists('sample_data.csv'):
        if analyzer.load_data('sample_data.csv'):
            results = analyzer.perform_complete_analysis()
            if results:
                return jsonify(results)
    
    return jsonify({'error': 'Sample data not found'}), 404

# HTML Template (to be saved as templates/index.html)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .file-input {
            margin: 10px;
            padding: 10px;
        }
        .btn {
            background: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .results {
            display: none;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Analysis Dashboard</h1>
            <p>Upload your CSV file for comprehensive analysis</p>
        </div>
        
        <div class="upload-section">
            <h3>Upload CSV File</h3>
            <p>File should have 'Timestamp' and 'Values' columns</p>
            <input type="file" id="csvFile" accept=".csv" class="file-input">
            <br>
            <button onclick="uploadFile()" class="btn">Analyze Data</button>
            <button onclick="analyzeSample()" class="btn">Use Sample Data</button>
        </div>
        
        <div id="loading" class="loading">
            <p>Processing your data...</p>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
        
        <div id="results" class="results">
            <h2>Analysis Results</h2>
            
            <div id="statistics">
                <h3>Statistics Summary</h3>
                <div id="stats-grid" class="stats-grid"></div>
            </div>
            
            <div id="plots">
                <div class="plot-container">
                    <h3>Time Series with Moving Averages</h3>
                    <img id="timeseries-plot" alt="Time Series Plot">
                </div>
                
                <div class="plot-container">
                    <h3>Data Distribution</h3>
                    <img id="histogram-plot" alt="Histogram Plot">
                </div>
                
                <div class="plot-container">
                    <h3>Peak and Low Detection</h3>
                    <img id="peaks-plot" alt="Peaks Plot">
                </div>
                
                <div class="plot-container" id="daily-plot-container">
                    <h3>Values on 12th Day</h3>
                    <img id="daily-plot" alt="Daily Plot">
                </div>
            </div>
            
            <div id="tables">
                <h3>Local Peaks</h3>
                <div id="peaks-table"></div>
                
                <h3>Local Lows</h3>
                <div id="lows-table"></div>
                
                <h3>Values Below 20</h3>
                <div id="below-table"></div>
                
                <h3>Accelerating Downward Trends</h3>
                <div id="accel-table"></div>
            </div>
        </div>
    </div>

    <script>
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        function displayResults(data) {
            // Display statistics
            const statsGrid = document.getElementById('stats-grid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.min.toFixed(2)}</div>
                    <div>Minimum</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.max.toFixed(2)}</div>
                    <div>Maximum</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.mean.toFixed(2)}</div>
                    <div>Average</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.median.toFixed(2)}</div>
                    <div>Median</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.std.toFixed(2)}</div>
                    <div>Std Dev</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.statistics.count}</div>
                    <div>Data Points</div>
                </div>
            `;

            // Display plots
            document.getElementById('timeseries-plot').src = 'data:image/png;base64,' + data.plots.timeseries;
            document.getElementById('histogram-plot').src = 'data:image/png;base64,' + data.plots.histogram;
            document.getElementById('peaks-plot').src = 'data:image/png;base64,' + data.plots.peaks;
            
            if (data.plots.daily) {
                document.getElementById('daily-plot').src = 'data:image/png;base64,' + data.plots.daily;
                document.getElementById('daily-plot-container').style.display = 'block';
            } else {
                document.getElementById('daily-plot-container').style.display = 'none';
            }

            // Display tables
            document.getElementById('peaks-table').innerHTML = createTable(data.peaks, ['Timestamp', 'Values']);
            document.getElementById('lows-table').innerHTML = createTable(data.lows, ['Timestamp', 'Values']);
            document.getElementById('below-table').innerHTML = createTable(data.below_20, ['Timestamp', 'Values']);
            document.getElementById('accel-table').innerHTML = createTable(data.accelerating_down, ['Timestamp', 'Values', 'diff', 'slope_accel']);

            document.getElementById('results').style.display = 'block';
        }

        function createTable(data, columns) {
            if (!data || data.length === 0) {
                return '<p>No data available</p>';
            }

            let html = '<table><thead><tr>';
            columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            html += '</tr></thead><tbody>';

            data.forEach(row => {
                html += '<tr>';
                columns.forEach(col => {
                    let value = row[col];
                    if (typeof value === 'number') {
                        value = value.toFixed(4);
                    }
                    html += `<td>${value || ''}</td>`;
                });
                html += '</tr>';
            });

            html += '</tbody></table>';
            return html;
        }

        function uploadFile() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Please select a CSV file');
                return;
            }

            showLoading();
            
            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Error uploading file: ' + error.message);
            });
        }

        function analyzeSample() {
            showLoading();
            
            fetch('/analyze_sample')
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Error analyzing sample data: ' + error.message);
            });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Create templates directory and save HTML template
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    print("Starting Data Analysis Dashboard...")
    print("Make sure your CSV file has 'Timestamp' and 'Values' columns")
    print("Access the dashboard at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)