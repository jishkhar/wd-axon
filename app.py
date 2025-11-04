import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import main as sim_main
from src import config


st.set_page_config(
    page_title="Project Axon - WD Innovation Bazaar 2025",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def run_simulation(dataset_size, vector_dim, query_count, top_k, demo_mode=False):
    """Runs the simulation using the existing project logic.

    This function updates `src.config` at runtime so the underlying
    modules use the UI-provided values. It then calls the helper
    functions in `main.py` and returns the benchmark results and the
    path to the generated image.
    """
    # Update runtime configuration
    config.DATASET_SIZE = dataset_size
    config.VECTOR_DIMENSIONS = vector_dim
    config.QUERY_COUNT = query_count
    config.TOP_K = top_k

    # Run stages and capture timings for a better UX
    t0 = time.perf_counter()

    vectors_data, query_data = sim_main.prepare_data(dataset_size, vector_dim)

    # Traditional system setup
    trad_msg = st.empty()
    trad_msg.info("Setting up Traditional System (building index and writing index file)...")
    traditional_ssd, index_file_name = sim_main.setup_traditional_system(vectors_data)
    trad_msg.success("Traditional system ready")

    # Axon system setup
    axon_msg = st.empty()
    axon_msg.info("Setting up Axon System (writing vectors to drive)...")
    axon_ssd = sim_main.setup_axon_system(vectors_data)
    axon_msg.success("Axon system ready")

    # Host and benchmark
    host = sim_main.Host()
    bench_msg = st.empty()
    bench_msg.info("Running benchmark (this may take a while)...")
    results = sim_main.run_benchmark(host, query_data, traditional_ssd, index_file_name, axon_ssd)
    bench_msg.success("Benchmark complete")

    # Generate plots (this also writes benchmark_results.png)
    plot_msg = st.empty()
    plot_msg.info("Generating comparison graphs...")
    sim_main.plot_results(results)
    plot_msg.success("Graphs generated")

    elapsed = time.perf_counter() - t0
    return results, os.path.abspath("benchmark_results.png"), elapsed


def pretty_results(results):
    """Return a dict of user-friendly metrics for display."""
    return {
        "Average Latency (Traditional) ms": results['trad_latency_avg'],
        "Average Latency (Axon) ms": results['axon_latency_avg'],
        "PCIe Traffic / Query (Traditional) MB": results['trad_traffic_avg'] * 1024,
        "PCIe Traffic / Query (Axon) MB": results['axon_traffic_avg'] * 1024,
        "Host CPU Work / Query (Traditional) ms": results['trad_cpu_avg'],
        "Host CPU Work / Query (Axon) ms": results['axon_cpu_avg'],
    }


def create_interactive_charts(results):
    """Create interactive Plotly charts for better visualization."""
    metrics = pretty_results(results)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Query Latency Comparison', 'Data Traffic Comparison', 'Host CPU Load Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # WD Brand Colors (approximate)
    wd_blue = '#0066CC'
    wd_orange = '#FF6B35'
    trad_color = '#FF6347'
    axon_color = '#4169E1'
    
    # Chart 1: Latency
    fig.add_trace(
        go.Bar(
            name='Traditional',
            x=['Latency (ms)'],
            y=[metrics['Average Latency (Traditional) ms']],
            marker_color=trad_color,
            text=[f"{metrics['Average Latency (Traditional) ms']:.4f}"],
            textposition='outside',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name='Project Axon',
            x=['Latency (ms)'],
            y=[metrics['Average Latency (Axon) ms']],
            marker_color=axon_color,
            text=[f"{metrics['Average Latency (Axon) ms']:.4f}"],
            textposition='outside',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Chart 2: Traffic
    fig.add_trace(
        go.Bar(
            name='Traditional',
            x=['Traffic (MB)'],
            y=[metrics['PCIe Traffic / Query (Traditional) MB']],
            marker_color=trad_color,
            text=[f"{metrics['PCIe Traffic / Query (Traditional) MB']:.4f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name='Project Axon',
            x=['Traffic (MB)'],
            y=[metrics['PCIe Traffic / Query (Axon) MB']],
            marker_color=axon_color,
            text=[f"{metrics['PCIe Traffic / Query (Axon) MB']:.4f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Chart 3: CPU
    fig.add_trace(
        go.Bar(
            name='Traditional',
            x=['CPU Time (ms)'],
            y=[metrics['Host CPU Work / Query (Traditional) ms']],
            marker_color=trad_color,
            text=[f"{metrics['Host CPU Work / Query (Traditional) ms']:.4f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(
            name='Project Axon',
            x=['CPU Time (ms)'],
            y=[metrics['Host CPU Work / Query (Axon) ms']],
            marker_color=axon_color,
            text=[f"{metrics['Host CPU Work / Query (Axon) ms']:.4f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=3)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text="Performance Metrics Comparison (Log Scale)",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig


def calculate_improvements(results):
    """Calculate percentage improvements."""
    metrics = pretty_results(results)
    
    latency_improvement = ((metrics['Average Latency (Traditional) ms'] - 
                           metrics['Average Latency (Axon) ms']) / 
                          metrics['Average Latency (Traditional) ms']) * 100
    
    traffic_improvement = ((metrics['PCIe Traffic / Query (Traditional) MB'] - 
                           metrics['PCIe Traffic / Query (Axon) MB']) / 
                          metrics['PCIe Traffic / Query (Traditional) MB']) * 100
    
    cpu_improvement = ((metrics['Host CPU Work / Query (Traditional) ms'] - 
                       metrics['Host CPU Work / Query (Axon) ms']) / 
                      metrics['Host CPU Work / Query (Traditional) ms']) * 100
    
    return {
        'latency': latency_improvement,
        'traffic': traffic_improvement,
        'cpu': cpu_improvement
    }


def main():
    # Custom CSS for WD branding
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #0066CC 0%, #004C99 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .main-header h1 {
            color: white !important;
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }
        .main-header p {
            color: #E0E0E0 !important;
            font-size: 1.2rem !important;
            margin: 0.5rem 0 0 0 !important;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #0066CC;
            margin: 1rem 0;
        }
        .improvement-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0 2rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>⚡ Project Axon</h1>
            <p>Next-Generation Computational Storage Solution</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'elapsed' not in st.session_state:
        st.session_state.elapsed = None
    if 'bunch_results' not in st.session_state:
        st.session_state.bunch_results = None

    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Results", "ℹ️ About"])

    with tab1:
        st.markdown("### 🎯 Benchmark Configuration")
        st.markdown("Configure your simulation parameters and run the performance comparison.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        preset = st.selectbox(
            "📋 Quick Preset",
            ["Default (50k vectors)", "Quick demo (2k)", "Large Scale (100k)", "Small Test (500)"],
            help="Choose a preset configuration or customize parameters below"
        )
        if preset.startswith("Quick demo"):
            default_dataset = 2000
            default_queries = 20
        elif preset.startswith("Large Scale"):
            default_dataset = 100000
            default_queries = 200
        elif preset.startswith("Small Test"):
            default_dataset = 500
            default_queries = 10
        else:
            default_dataset = config.DATASET_SIZE
            default_queries = config.QUERY_COUNT

        st.markdown("#### Simulation Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dataset_size = st.number_input(
                "🗄️ Dataset size (vectors)",
                min_value=100,
                value=int(default_dataset),
                step=100,
                help="Number of vectors to store in the database"
            )
        with col2:
            vector_dim = st.number_input(
                "📐 Vector dimensions",
                min_value=8,
                value=int(config.VECTOR_DIMENSIONS),
                step=8,
                help="Dimensionality of each vector (e.g., 128 for image embeddings)"
            )
        with col3:
            query_count = st.number_input(
                "🔍 Number of queries",
                min_value=1,
                value=int(default_queries),
                step=1,
                help="How many search queries to run for the benchmark"
            )
        with col4:
            top_k = st.number_input(
                "🎯 Top K results",
                min_value=1,
                value=int(config.TOP_K),
                step=1,
                help="Number of nearest neighbors to retrieve per query"
            )

        # Calculate estimated resources
        est_size_gb = (dataset_size * vector_dim * 4) / (1024**3)
        st.info(f"📊 **Estimated Data Size:** {est_size_gb:.2f} GB | **Estimated Runtime:** ~{query_count * 0.1:.1f}s (varies by hardware)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            run_btn = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)
        with col_btn2:
            run_bunch_btn = st.button("📊 Run Bunch (10x)", type="secondary", use_container_width=True)

        if run_btn:
            try:
                with st.spinner("⚙️ Running simulation — this may take several minutes for large datasets..."):
                    results, image_path, elapsed = run_simulation(dataset_size, vector_dim, query_count, top_k)

                # Store results in session state
                st.session_state.results = results
                st.session_state.image_path = image_path
                st.session_state.metrics = pretty_results(results)
                st.session_state.elapsed = elapsed
                st.session_state.improvements = calculate_improvements(results)

                st.success(f"✅ Benchmark completed successfully in {elapsed:.1f}s — Check the **Results** tab!")

            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                st.exception(e)
        
        if run_bunch_btn:
            try:
                # Define 10 different configurations
                bunch_configs = [
                    {"dataset": 1000, "dim": 64, "queries": 10, "k": 3},
                    {"dataset": 2000, "dim": 128, "queries": 20, "k": 5},
                    {"dataset": 5000, "dim": 128, "queries": 50, "k": 5},
                    {"dataset": 10000, "dim": 256, "queries": 50, "k": 10},
                    {"dataset": 20000, "dim": 128, "queries": 100, "k": 5},
                    {"dataset": 30000, "dim": 256, "queries": 100, "k": 10},
                    {"dataset": 50000, "dim": 128, "queries": 100, "k": 5},
                    {"dataset": 50000, "dim": 256, "queries": 150, "k": 10},
                    {"dataset": 75000, "dim": 128, "queries": 100, "k": 5},
                    {"dataset": 100000, "dim": 128, "queries": 200, "k": 5},
                ]
                
                bunch_results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                last_results = None
                last_image_path = None
                total_elapsed = 0
                
                for idx, cfg in enumerate(bunch_configs):
                    status_text.info(f"⚙️ Running benchmark {idx + 1}/10: {cfg['dataset']} vectors, {cfg['dim']} dims, {cfg['queries']} queries, top-{cfg['k']}")
                    
                    results, image_path, elapsed = run_simulation(
                        cfg['dataset'],
                        cfg['dim'],
                        cfg['queries'],
                        cfg['k']
                    )
                    
                    # Store the last result for display in Results tab
                    last_results = results
                    last_image_path = image_path
                    total_elapsed += elapsed
                    
                    metrics = pretty_results(results)
                    improvements = calculate_improvements(results)
                    
                    bunch_results_list.append({
                        "Dataset Size": cfg['dataset'],
                        "Dimensions": cfg['dim'],
                        "Queries": cfg['queries'],
                        "Top K": cfg['k'],
                        "Trad Latency (ms)": f"{metrics['Average Latency (Traditional) ms']:.6f}",
                        "Axon Latency (ms)": f"{metrics['Average Latency (Axon) ms']:.6f}",
                        "Latency Improvement (%)": f"{improvements['latency']:.2f}",
                        "Trad Traffic (MB)": f"{metrics['PCIe Traffic / Query (Traditional) MB']:.6f}",
                        "Axon Traffic (MB)": f"{metrics['PCIe Traffic / Query (Axon) MB']:.6f}",
                        "Traffic Improvement (%)": f"{improvements['traffic']:.2f}",
                        "Trad CPU (ms)": f"{metrics['Host CPU Work / Query (Traditional) ms']:.6f}",
                        "Axon CPU (ms)": f"{metrics['Host CPU Work / Query (Axon) ms']:.6f}",
                        "CPU Improvement (%)": f"{improvements['cpu']:.2f}",
                        "Runtime (s)": f"{elapsed:.2f}"
                    })
                    
                    progress_bar.progress((idx + 1) / 10)
                
                # Store both bunch results and last individual result
                st.session_state.bunch_results = bunch_results_list
                st.session_state.results = last_results
                st.session_state.image_path = last_image_path
                st.session_state.metrics = pretty_results(last_results)
                st.session_state.elapsed = total_elapsed
                st.session_state.improvements = calculate_improvements(last_results)
                
                progress_bar.empty()
                status_text.success(f"✅ All 10 benchmarks completed in {total_elapsed:.1f}s! Check the **Results** tab for the comprehensive table.")
                
            except Exception as e:
                st.error(f"❌ Bunch simulation failed: {e}")
                st.exception(e)

    with tab2:
        if st.session_state.results is None:
            st.info("👈 Run a benchmark in the **Configuration** tab to see results here.")
            
            # Show placeholder content
            st.markdown("### What You'll See Here:")
            st.markdown("""
            - **Performance Metrics**: Detailed comparison of latency, traffic, and CPU usage
            - **Interactive Charts**: Visual representation of performance gains
            - **Improvement Analysis**: Percentage improvements achieved by Project Axon
            - **Downloadable Reports**: Export results and graphs for presentations
            """)
        else:
            st.markdown(f"### 📊 Benchmark Results")
            st.success(f"✅ Completed in **{st.session_state.elapsed:.1f} seconds**")

            metrics = st.session_state.metrics
            improvements = st.session_state.improvements

            # Key Improvements Banner
            st.markdown("### 🎯 Key Performance Improvements")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #0066CC; margin: 0;">⚡ Latency Reduction</h4>
                    <h2 style="color: #28a745; margin: 0.5rem 0;">{improvements['latency']:.1f}%</h2>
                    <p style="margin: 0; color: #666;">Faster query response time</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #0066CC; margin: 0;">📉 Traffic Reduction</h4>
                    <h2 style="color: #28a745; margin: 0.5rem 0;">{improvements['traffic']:.1f}%</h2>
                    <p style="margin: 0; color: #666;">Less PCIe bandwidth used</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #0066CC; margin: 0;">🖥️ CPU Offload</h4>
                    <h2 style="color: #28a745; margin: 0.5rem 0;">{improvements['cpu']:.1f}%</h2>
                    <p style="margin: 0; color: #666;">Reduced host CPU load</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Plotly Charts
            st.markdown("### 📈 Performance Comparison")
            fig = create_interactive_charts(st.session_state.results)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Metrics
            st.markdown("### 📋 Detailed Metrics")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 🔴 Traditional System")
                st.metric("Average Latency", f"{metrics['Average Latency (Traditional) ms']:.6f} ms")
                st.metric("PCIe Traffic per Query", f"{metrics['PCIe Traffic / Query (Traditional) MB']:.6f} MB")
                st.metric("Host CPU Time per Query", f"{metrics['Host CPU Work / Query (Traditional) ms']:.6f} ms")
            
            with col_b:
                st.markdown("#### 🔵 Project Axon")
                st.metric(
                    "Average Latency",
                    f"{metrics['Average Latency (Axon) ms']:.6f} ms",
                    delta=f"-{improvements['latency']:.1f}%",
                    delta_color="inverse"
                )
                st.metric(
                    "PCIe Traffic per Query",
                    f"{metrics['PCIe Traffic / Query (Axon) MB']:.6f} MB",
                    delta=f"-{improvements['traffic']:.1f}%",
                    delta_color="inverse"
                )
                st.metric(
                    "Host CPU Time per Query",
                    f"{metrics['Host CPU Work / Query (Axon) ms']:.6f} ms",
                    delta=f"-{improvements['cpu']:.1f}%",
                    delta_color="inverse"
                )

            # Full results table
            st.markdown("### 📊 Complete Data Table")
            st.dataframe(
                {k: [v] for k, v in metrics.items()},
                use_container_width=True
            )
            
            # Display bunch results if available
            if st.session_state.bunch_results is not None:
                st.markdown("### 📊 Bunch Benchmark Results (10 Configurations)")
                st.info("This table shows results from 10 different benchmark configurations run in batch mode.")
                
                bunch_df = pd.DataFrame(st.session_state.bunch_results)
                st.dataframe(bunch_df, use_container_width=True, height=400)
                
                # Download button for bunch results
                csv = bunch_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Bunch Results (CSV)",
                    csv,
                    file_name="project_axon_bunch_results.csv",
                    mime="text/csv",
                    use_container_width=False
                )

            # Matplotlib image and download
            if st.session_state.image_path and os.path.exists(st.session_state.image_path):
                st.markdown("### 📸 Static Comparison Graph")
                st.image(st.session_state.image_path, use_container_width=True)

                with open(st.session_state.image_path, "rb") as f:
                    img_bytes = f.read()
                st.download_button(
                    "📥 Download Benchmark Image",
                    img_bytes,
                    file_name="project_axon_benchmark_results.png",
                    mime="image/png",
                    use_container_width=False
                )
    
    with tab3:
        st.markdown("### ℹ️ About Project Axon")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Project Axon** represents the next evolution in storage technology—**Computational Storage Devices (CSDs)** 
            that process data where it lives, dramatically reducing data movement and improving system efficiency.
            
            #### 🎯 The Challenge
            Traditional storage architectures require massive amounts of data to be transferred from SSDs to the host CPU 
            for processing. For vector similarity search and AI workloads, this creates:
            - High PCIe bandwidth consumption
            - Increased latency
            - CPU bottlenecks
            - Power inefficiency
            
            #### 💡 The Axon Solution
            By embedding computational capabilities directly into the SSD, Project Axon:
            - ✅ Processes queries **on-device** using internal vector search engines
            - ✅ Transfers only **query results**, not entire datasets
            - ✅ Offloads work from the host CPU
            - ✅ Reduces latency by **orders of magnitude**
            
            #### 🚀 Use Cases
            - **AI/ML Inference**: Semantic search, recommendation systems
            - **Database Acceleration**: Analytics on large datasets
            - **Content Delivery**: Fast similarity matching for media
            - **Genomics**: DNA sequence analysis
            
            #### 🏆 Innovation Bazaar 2025
            This demo showcases a realistic simulation of Project Axon's performance benefits 
            compared to traditional storage architectures, using vector similarity search as a benchmark workload.
            """)
        
        with col2:
            st.markdown("#### 📊 Technology Stack")
            st.code("""
• Python 3.x
• Streamlit (UI)
• HNSW (Vector Index)
• NumPy (Computation)
• Plotly (Visualization)
            """)
            
            st.markdown("#### 🔗 Quick Stats")
            st.info("""
            **Simulation Features:**
            - Configurable datasets
            - Real HNSW indexing
            - PCIe bandwidth modeling
            - CPU cost estimation
            - Interactive visualization
            """)
            
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>Presented by the Axon Team</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
