# 🚀 Project Axon - Computational Storage Demo

**Western Digital Innovation Bazaar 2025**

---

## 🎯 Overview

Project Axon demonstrates the revolutionary potential of **Computational Storage Devices (CSDs)** — storage that doesn't just store data, but processes it in-place. This interactive demo compares traditional host-based vector search with Axon's on-device computational approach.

## ✨ Features

- **Interactive Streamlit UI** - Professional, polished interface suitable for presentations
- **Real-time Benchmarking** - Compare traditional vs. Axon architecture with configurable parameters
- **Interactive Visualizations** - Plotly charts showing performance improvements
- **Multiple Presets** - Quick demo, default, large scale, and small test configurations
- **Detailed Analytics** - Latency, PCIe traffic, and CPU usage metrics
- **Export Capabilities** - Download benchmark results and graphs

## 🏆 Key Performance Improvements

Project Axon typically demonstrates:
- **90%+** reduction in query latency
- **99%+** reduction in PCIe traffic
- **99%+** reduction in host CPU load

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
cd wd-axon

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How to Use

### 1. Configuration Tab
- Select a preset (Quick demo recommended for first run)
- Adjust parameters:
  - **Dataset size**: Number of vectors to index
  - **Vector dimensions**: Dimensionality of each vector
  - **Number of queries**: How many searches to run
  - **Top K results**: Number of nearest neighbors to find
- Click **"🚀 Run Benchmark"**

### 2. Results Tab
- View performance improvements (latency, traffic, CPU)
- Explore interactive Plotly charts
- Compare detailed metrics side-by-side
- Download graphs for presentations

### 3. About Tab
- Learn about Project Axon technology
- Understand the problem and solution
- Explore use cases

## 🎨 UI Features

- **WD Brand Colors** - Professional blue gradient header
- **Responsive Layout** - Wide layout optimized for presentations
- **Interactive Charts** - Zoomable, hoverable Plotly visualizations
- **Real-time Progress** - Spinners and status updates during benchmarks
- **Session Persistence** - Results saved across tab switches
- **Celebration Effects** - Success animations on completion

## 📁 Project Structure

```
wd-axon/
├── app.py                 # Main Streamlit application
├── main.py               # Core simulation logic
├── requirements.txt      # Python dependencies
├── README_APP.md        # This file
├── src/
│   ├── config.py        # Simulation parameters
│   ├── host.py          # Host computer simulation
│   ├── standard_ssd.py  # Traditional SSD simulation
│   └── axon_ssd.py      # Axon CSD simulation
```

## 🔧 Configuration Options

### Presets
- **Quick demo (2k)**: Fast testing, 2,000 vectors, 20 queries
- **Default (50k)**: Standard benchmark, 50,000 vectors, 100 queries
- **Large Scale (100k)**: Heavy workload, 100,000 vectors, 200 queries
- **Small Test (500)**: Ultra-fast, 500 vectors, 10 queries

### Custom Parameters
All parameters can be manually adjusted regardless of preset selection.

## 💡 Tips for Presentation

1. **Start with Quick Demo** - Shows results in seconds
2. **Explain the Charts** - Interactive Plotly charts can be zoomed and explored
3. **Highlight the Improvements** - Green percentage badges show gains clearly
4. **Show the About Tab** - Provides context for the innovation
5. **Export Results** - Download graphs for slides or reports

## 🛠️ Technical Details

### Simulation Components
- **HNSW Indexing**: Real approximate nearest neighbor search
- **PCIe Modeling**: Realistic bandwidth and latency calculations
- **CPU Cost Estimation**: Host processing time simulation

### Technologies Used
- **Streamlit**: Interactive web application framework
- **NumPy**: Numerical computing and vector operations
- **HNSW**: High-performance vector similarity search
- **Plotly**: Interactive data visualization
- **Matplotlib**: Static graph generation

## 🎯 Use Cases Demonstrated

- **AI/ML Inference**: Semantic search in embedding spaces
- **Recommendation Systems**: Finding similar items/users
- **Image Search**: Visual similarity matching
- **Database Analytics**: Fast aggregations on large datasets

## 📈 Understanding the Results

### Latency (ms)
- **Traditional**: Host reads entire index, performs search
- **Axon**: On-device search, returns only results
- **Impact**: Faster response times, better user experience

### PCIe Traffic (MB)
- **Traditional**: Transfers full index file over PCIe
- **Axon**: Transfers only tiny query and small results
- **Impact**: Reduced bandwidth contention, system scalability

### CPU Usage (ms)
- **Traditional**: Host CPU performs all search computation
- **Axon**: CSD handles search, host CPU is free
- **Impact**: More CPU available for other tasks

## 🚨 Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Memory Issues
- Use smaller dataset sizes
- Start with "Quick demo" preset
- Close other applications

### Slow Performance
- Reduce number of queries
- Use "Small Test" preset
- Check system resources

## 📝 License

This is a demonstration project for Western Digital's Innovation Bazaar 2025.

## 👥 Credits

Developed by the Project Axon Team for WD Innovation Bazaar 2025.

---

**Ready to revolutionize storage? Run the demo and see the future!** ⚡
