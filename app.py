#!/usr/bin/env python3
"""
Digital Sports Injury Autopsy Engine 🏥📉 - Streamlit App
A forensic analysis system for athletic injuries using local LLM via Ollama
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Digital Sports Injury Autopsy Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10B981;
        margin-bottom: 1rem;
    }
    .analysis-card {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E0F2FE;
        margin-bottom: 1.5rem;
        white-space: pre-wrap;
        font-family: 'Courier New', monospace;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class AthleteMetrics:
    """Data class for athlete metrics"""
    date: str
    training_load: float
    sleep_hours: float
    sleep_quality: float
    nutrition_score: float
    stress_level: float
    recovery_score: float
    acute_load: float
    chronic_load: float
    acwr: float
    injury_occurred: bool = False
    injury_type: Optional[str] = None
    injury_severity: Optional[str] = None
    biomechanical_notes: Optional[str] = None
    previous_injuries: Optional[str] = None

class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                return models
            return []
        except:
            return []
    
    def generate(self, prompt: str, system_prompt: str = None, 
                temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """Generate response from LLM"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "You are a sports medicine expert providing detailed analysis.",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Connection error: {str(e)}"

class InjuryAutopsyEngine:
    """Main engine for forensic injury analysis"""
    
    def __init__(self):
        self.metrics_data = []
        self.injury_date = None
        self.injury_details = {}
        self.analysis_results = {
            "early_warnings": [],
            "root_causes": [],
            "prevention_recommendations": [],
            "risk_factors": {},
            "biomechanical_insights": [],
            "prevention_plan": ""
        }
        
    def generate_sample_data(self, athlete_name: str, sport: str, days: int = 90):
        """Generate synthetic athlete data"""
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Simulate training cycles
            cycle_day = i % 21
            
            if cycle_day < 14:
                training_load = 60 + 10 * np.sin(cycle_day * 0.5) + np.random.normal(0, 5)
            else:
                training_load = 40 + 5 * np.sin(cycle_day * 0.3) + np.random.normal(0, 3)
            
            # Simulate injury (day 70)
            injury_occurred = (i == 70)
            injury_type = "Hamstring Strain" if injury_occurred else None
            injury_severity = "Grade 2" if injury_occurred else None
            
            # Generate correlated metrics
            sleep_hours = 8.0 - (training_load - 60) * 0.03 + np.random.normal(0, 0.5)
            sleep_quality = max(1, min(10, 8 - (training_load - 60) * 0.05 + np.random.normal(0, 1)))
            nutrition_score = max(1, min(10, 7 - (training_load - 60) * 0.03 + np.random.normal(0, 1)))
            stress_level = min(10, 3 + (training_load - 60) * 0.1 + np.random.normal(0, 1))
            recovery_score = max(1, min(10, 8 - (training_load - 60) * 0.08 + np.random.normal(0, 1)))
            
            # Calculate loads
            if i >= 28:
                acute_load = training_load
                chronic_load = training_load * 0.9
                acwr = acute_load / chronic_load if chronic_load > 0 else 1.0
            else:
                acute_load = training_load
                chronic_load = training_load
                acwr = 1.0
            
            # Add biomechanical notes for injury day
            biomechanical_notes = None
            if injury_occurred:
                biomechanical_notes = "Reduced hip extension, increased anterior pelvic tilt noted during sprints"
                self.injury_date = date_str
                self.injury_details = {
                    "type": injury_type,
                    "severity": injury_severity,
                    "date": date_str
                }
            
            metrics = AthleteMetrics(
                date=date_str,
                training_load=max(20, min(100, training_load)),
                sleep_hours=max(4, min(10, sleep_hours)),
                sleep_quality=max(1, min(10, sleep_quality)),
                nutrition_score=max(1, min(10, nutrition_score)),
                stress_level=max(1, min(10, stress_level)),
                recovery_score=max(1, min(10, recovery_score)),
                acute_load=max(20, min(100, acute_load)),
                chronic_load=max(20, min(100, chronic_load)),
                acwr=acwr,
                injury_occurred=injury_occurred,
                injury_type=injury_type,
                injury_severity=injury_severity,
                biomechanical_notes=biomechanical_notes,
                previous_injuries="Left hamstring strain 8 months ago" if i > 30 else None
            )
            
            self.metrics_data.append(metrics)
        
        return True
    
    def get_dataframe(self):
        """Convert metrics data to DataFrame"""
        df = pd.DataFrame([asdict(m) for m in self.metrics_data])
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def create_visualizations(self):
        """Create interactive visualizations"""
        df = self.get_dataframe()
        
        # Create simpler visualization - single comprehensive figure
        fig = go.Figure()
        
        # Add traces for different metrics
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['training_load'],
            mode='lines',
            name='Training Load',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['recovery_score'] * 10,  # Scale for visibility
            mode='lines',
            name='Recovery Score (x10)',
            line=dict(color='green', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['sleep_hours'],
            mode='lines',
            name='Sleep Hours',
            line=dict(color='purple', width=1.5),
            yaxis='y3'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['acwr'],
            mode='lines',
            name='ACWR',
            line=dict(color='orange', width=2),
            yaxis='y4'
        ))
        
        # Add injury markers if any
        injury_df = df[df['injury_occurred']]
        if not injury_df.empty:
            fig.add_trace(go.Scatter(
                x=injury_df['date'],
                y=[df['training_load'].max()] * len(injury_df),
                mode='markers',
                name='Injury',
                marker=dict(color='red', size=12, symbol='x'),
                yaxis='y'
            ))
        
        # Add ACWR threshold lines
        fig.add_hline(y=1.5, line_dash="dot", line_color="red", 
                     annotation_text="High Risk Threshold", row=None, col=None,
                     yref="y4")
        fig.add_hline(y=0.8, line_dash="dot", line_color="green", 
                     annotation_text="Optimal Zone", row=None, col=None,
                     yref="y4")
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Athlete Metrics Dashboard - Training Load, Recovery, Sleep & ACWR",
            xaxis=dict(
                title="Date",
                tickformat="%b %d"
            ),
            yaxis=dict(
                title="Training Load",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                side="left",
                position=0
            ),
            yaxis2=dict(
                title="Recovery Score (x10)",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                overlaying="y",
                side="right",
                position=1
            ),
            yaxis3=dict(
                title="Sleep Hours",
                titlefont=dict(color="purple"),
                tickfont=dict(color="purple"),
                overlaying="y",
                side="right",
                position=0.85,
                anchor="free"
            ),
            yaxis4=dict(
                title="ACWR",
                titlefont=dict(color="orange"),
                tickfont=dict(color="orange"),
                overlaying="y",
                side="right",
                position=0.7
            ),
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        df = self.get_dataframe()
        
        # Select numeric columns for correlation
        numeric_cols = ['training_load', 'sleep_hours', 'sleep_quality', 
                       'nutrition_score', 'stress_level', 'recovery_score', 'acwr']
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Metrics Correlation Heatmap"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_weekly_trends(self):
        """Create weekly trend analysis"""
        df = self.get_dataframe()
        
        # Add week number
        df['week'] = df['date'].dt.isocalendar().week
        
        # Group by week
        weekly_avg = df.groupby('week').agg({
            'training_load': 'mean',
            'recovery_score': 'mean',
            'sleep_hours': 'mean',
            'acwr': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weekly_avg['week'],
            y=weekly_avg['training_load'],
            mode='lines+markers',
            name='Training Load',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_avg['week'],
            y=weekly_avg['recovery_score'] * 10,
            mode='lines+markers',
            name='Recovery Score (x10)',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add injury week marker
        if self.injury_date:
            injury_week = pd.to_datetime(self.injury_date).isocalendar().week
            fig.add_vline(
                x=injury_week,
                line_dash="dash",
                line_color="red",
                annotation_text="Injury Week"
            )
        
        fig.update_layout(
            title="Weekly Trends Analysis",
            xaxis_title="Week Number",
            yaxis_title="Metrics",
            hovermode='x unified',
            height=400,
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        # First check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Then check if service is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return True, ["llama3.2", "mistral", "orca-mini"]  # Return default models
    except:
        pass
    return False, []

def main():
    """Main Streamlit application"""
    
    # App header
    st.markdown('<div class="main-header">🏥 Digital Sports Injury Autopsy Engine</div>', 
                unsafe_allow_html=True)
    st.markdown("**A forensic analysis system for athletic injuries using local LLM via Ollama**")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'engine' not in st.session_state:
        st.session_state.engine = InjuryAutopsyEngine()
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Athlete information
        athlete_name = st.text_input("Athlete Name", value="Alex Johnson")
        sport = st.selectbox("Sport", 
                           ["Soccer", "Basketball", "Football", "Running", 
                            "Swimming", "Tennis", "Volleyball", "Other"])
        
        # Ollama configuration
        st.markdown("### 🤖 LLM Configuration")
        ollama_installed, available_models = check_ollama_installation()
        
        if ollama_installed:
            st.success("✅ Ollama is installed")
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model", 
                available_models,
                index=available_models.index("llama3.2") if "llama3.2" in available_models else 0
            )
            
            # Test connection button
            if st.button("Test Ollama Connection"):
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Ollama service is running")
                    else:
                        st.error("❌ Ollama service not responding")
                except:
                    st.error("❌ Cannot connect to Ollama service")
                    
        else:
            st.error("❌ Ollama not detected")
            st.info("""
            **To use LLM features:**
            1. Install Ollama from https://ollama.com/
            2. Start Ollama: `ollama serve`
            3. Pull a model: `ollama pull llama3.2`
            """)
            selected_model = "llama3.2"
        
        # Analysis options
        st.markdown("### 📊 Analysis Options")
        analysis_days = st.slider("Days to analyze", 30, 180, 90)
        include_counterfactual = st.checkbox("Include Counterfactual Analysis", value=True)
        generate_prevention = st.checkbox("Generate Prevention Plan", value=True)
        
        # Run analysis button
        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Generating sample data..."):
                # Generate sample data
                st.session_state.engine.generate_sample_data(athlete_name, sport, analysis_days)
                
                # Initialize Ollama client if available
                if ollama_installed:
                    st.session_state.ollama_client = OllamaClient(model=selected_model)
                
                # Store for later use
                st.session_state.analysis_complete = True
                st.session_state.athlete_name = athlete_name
                st.session_state.sport = sport
                st.session_state.selected_model = selected_model
                st.session_state.ollama_available = ollama_installed
                
                st.success("✅ Analysis data generated successfully!")
    
    # Main content area
    if st.session_state.analysis_complete:
        engine = st.session_state.engine
        df = engine.get_dataframe()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Dashboard", "🔍 Analysis", "🛡️ Prevention", "📊 Metrics", "📄 Report"
        ])
        
        with tab1:
            st.markdown('<div class="sub-header">Athlete Metrics Dashboard</div>', 
                       unsafe_allow_html=True)
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Load (Avg)", f"{df['training_load'].mean():.1f}")
            with col2:
                st.metric("Recovery Score", f"{df['recovery_score'].mean():.1f}/10")
            with col3:
                acwr_color = "normal" if df['acwr'].mean() < 1.5 else "inverse"
                st.metric("ACWR (Avg)", f"{df['acwr'].mean():.2f}", delta_color=acwr_color)
            with col4:
                if engine.injury_date:
                    st.metric("Injury Date", engine.injury_date)
            
            # Main visualization
            st.plotly_chart(engine.create_visualizations(), use_container_width=True)
            
            # Additional visualizations in columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(engine.create_correlation_heatmap(), use_container_width=True)
            with col2:
                st.plotly_chart(engine.create_weekly_trends(), use_container_width=True)
        
        with tab2:
            st.markdown('<div class="sub-header">Forensic Analysis Results</div>', 
                       unsafe_allow_html=True)
            
            if st.session_state.ollama_available and st.session_state.ollama_client:
                # Analysis sections
                analysis_sections = st.container()
                
                with analysis_sections:
                    # Biomechanical Analysis
                    with st.expander("🧬 Biomechanical Analysis", expanded=True):
                        if st.button("Run Biomechanical Analysis", key="bio_analysis"):
                            with st.spinner("Analyzing biomechanics..."):
                                biomech_prompt = f"""
                                ROLE: Sports Biomechanist
                                TASK: Analyze biomechanical factors contributing to {engine.injury_details['type']}
                                
                                ATHLETE: {st.session_state.athlete_name}
                                SPORT: {st.session_state.sport}
                                INJURY: {engine.injury_details['type']} ({engine.injury_details['severity']})
                                INJURY DATE: {engine.injury_date}
                                
                                BIOMECHANICAL NOTES: {df[df['injury_occurred']]['biomechanical_notes'].iloc[0] if not df[df['injury_occurred']].empty else 'No specific notes'}
                                
                                PREVIOUS INJURIES: {df['previous_injuries'].iloc[-1] if not df.empty else 'None'}
                                
                                Analyze:
                                1. Likely biomechanical breakdown points
                                2. Movement pattern deviations
                                3. Muscle activation/coordination issues
                                4. Sport-specific biomechanical risk factors
                                
                                Provide structured analysis with actionable insights.
                                """
                                
                                analysis = st.session_state.ollama_client.generate(biomech_prompt)
                                engine.analysis_results["biomechanical_insights"].append(analysis)
                                st.markdown(f'<div class="analysis-card">{analysis}</div>', 
                                          unsafe_allow_html=True)
                    
                    # Physiotherapist Analysis
                    with st.expander("🩺 Physiotherapist Analysis"):
                        if st.button("Run Physiotherapist Analysis", key="physio_analysis"):
                            with st.spinner("Conducting physio assessment..."):
                                physio_prompt = f"""
                                ROLE: Sports Physiotherapist
                                TASK: Analyze tissue loading and recovery factors
                                
                                TRAINING LOAD TREND (last 28 days): {df['training_load'].tail(28).mean():.1f} avg
                                RECOVERY SCORES (last 14 days): {df['recovery_score'].tail(14).mean():.1f} avg
                                SLEEP HOURS (last 14 days): {df['sleep_hours'].tail(14).mean():.1f} avg
                                
                                INJURY HISTORY: {df['previous_injuries'].iloc[-1] if not df.empty else 'No previous injuries'}
                                
                                Analyze:
                                1. Tissue overload mechanisms for {engine.injury_details['type']}
                                2. Recovery-deficit contributions
                                3. Previous injury implications
                                4. Load management failures
                                5. Rehabilitation recommendations
                                
                                Focus on clinical reasoning and provide practical recommendations.
                                """
                                
                                analysis = st.session_state.ollama_client.generate(physio_prompt)
                                engine.analysis_results["root_causes"].append(analysis)
                                st.markdown(f'<div class="analysis-card">{analysis}</div>', 
                                          unsafe_allow_html=True)
                    
                    # Data Analyst Analysis
                    with st.expander("📊 Data Analyst Analysis"):
                        if st.button("Run Data Analysis", key="data_analysis"):
                            with st.spinner("Analyzing data patterns..."):
                                data_prompt = f"""
                                ROLE: Sports Data Analyst
                                TASK: Quantitative pattern detection
                                
                                DATA SUMMARY (last 30 days pre-injury):
                                - Training Load: Mean={df['training_load'].tail(30).mean():.1f}, Std={df['training_load'].tail(30).std():.1f}
                                - ACWR: Mean={df['acwr'].tail(30).mean():.2f}, Max={df['acwr'].tail(30).max():.2f}
                                - Recovery Score: Mean={df['recovery_score'].tail(30).mean():.1f}, Min={df['recovery_score'].tail(30).min():.1f}
                                - Sleep Hours: Mean={df['sleep_hours'].tail(30).mean():.1f}
                                
                                CORRELATIONS:
                                - Load vs Recovery: {df['training_load'].corr(df['recovery_score']):.2f}
                                - Load vs Sleep: {df['training_load'].corr(df['sleep_hours']):.2f}
                                - Stress vs Recovery: {df['stress_level'].corr(df['recovery_score']):.2f}
                                
                                Identify:
                                1. Statistical outliers and patterns
                                2. Metric correlations and thresholds
                                3. Predictive indicators of injury
                                4. Data-driven risk factors
                                
                                Present clear, data-driven insights with specific numbers.
                                """
                                
                                analysis = st.session_state.ollama_client.generate(data_prompt)
                                engine.analysis_results["risk_factors"]["quantitative"] = analysis
                                st.markdown(f'<div class="analysis-card">{analysis}</div>', 
                                          unsafe_allow_html=True)
                
                # Counterfactual Analysis
                if include_counterfactual:
                    with st.expander("🔮 Counterfactual Analysis"):
                        st.write("Explore 'what-if' scenarios to understand injury prevention")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            load_reduction = st.slider("Load Reduction (%)", 0, 30, 12, key="load_slider")
                        with col2:
                            sleep_increase = st.slider("Sleep Increase (hours)", 0.0, 2.0, 1.0, 0.1, key="sleep_slider")
                        with col3:
                            recovery_improvement = st.slider("Recovery Improvement (%)", 0, 40, 20, key="rec_slider")
                        
                        if st.button("Run Counterfactual Analysis", key="cf_analysis"):
                            with st.spinner("Running counterfactual analysis..."):
                                cf_prompt = f"""
                                ROLE: Sports Scientist
                                TASK: Counterfactual injury prevention analysis
                                
                                BASE SCENARIO:
                                - Athlete: {st.session_state.athlete_name}
                                - Injury: {engine.injury_details['type']} ({engine.injury_details['severity']})
                                - Date: {engine.injury_date}
                                - Pre-injury ACWR: {df[df['date'] == engine.injury_date]['acwr'].iloc[0] if not df[df['date'] == engine.injury_date].empty else 'N/A'}
                                
                                INTERVENTION SCENARIO:
                                - Weekly training load reduced by {load_reduction}%
                                - Sleep duration increased by {sleep_increase} hours/night
                                - Recovery protocols improved by {recovery_improvement}%
                                
                                Analyze:
                                1. Likely impact on injury risk probability
                                2. Required implementation changes
                                3. Expected timeline for effects
                                4. Secondary performance benefits
                                5. Practical implementation steps
                                
                                Provide evidence-based counterfactual reasoning with specific estimates.
                                """
                                
                                analysis = st.session_state.ollama_client.generate(cf_prompt)
                                engine.analysis_results["prevention_recommendations"].append({
                                    "intervention": {
                                        "load_reduction": load_reduction,
                                        "sleep_increase": sleep_increase,
                                        "recovery_improvement": recovery_improvement
                                    },
                                    "analysis": analysis
                                })
                                st.markdown(f'<div class="analysis-card">{analysis}</div>', 
                                          unsafe_allow_html=True)
            else:
                st.warning("⚠️ Ollama is not available. Please install and run Ollama to use analysis features.")
                st.info("The dashboard and metrics are still available. Install Ollama for LLM-powered analysis.")
        
        with tab3:
            st.markdown('<div class="sub-header">Injury Prevention Plan</div>', 
                       unsafe_allow_html=True)
            
            if st.session_state.ollama_available and st.session_state.ollama_client and generate_prevention:
                if st.button("🛡️ Generate Comprehensive Prevention Plan", type="primary"):
                    with st.spinner("Creating comprehensive prevention plan..."):
                        prevention_prompt = f"""
                        ROLE: Head of Sports Medicine
                        TASK: Create actionable injury prevention and rehabilitation plan
                        
                        ATHLETE: {st.session_state.athlete_name}
                        SPORT: {st.session_state.sport}
                        INJURY: {engine.injury_details['type']} ({engine.injury_details['severity']})
                        INJURY DATE: {engine.injury_date}
                        
                        KEY METRICS PRE-INJURY:
                        - Average Training Load: {df['training_load'].mean():.1f}
                        - Average Recovery Score: {df['recovery_score'].mean():.1f}/10
                        - Average ACWR: {df['acwr'].mean():.2f}
                        - Average Sleep: {df['sleep_hours'].mean():.1f} hours
                        
                        Create a comprehensive, structured prevention and rehabilitation plan with:
                        
                        PHASE 1: IMMEDIATE POST-INJURY (Days 1-7)
                        [List specific actions, treatments, and monitoring]
                        
                        PHASE 2: EARLY REHABILITATION (Weeks 2-4)
                        [List graduated exercises, load progression, and criteria]
                        
                        PHASE 3: RETURN TO TRAINING (Weeks 5-8)
                        [List sport-specific drills, load management, and testing]
                        
                        PHASE 4: PREVENTION SYSTEM (Months 3+)
                        [List ongoing monitoring, screening, and maintenance]
                        
                        MONITORING PROTOCOLS:
                        [List specific metrics, frequency, and thresholds]
                        
                        SUCCESS METRICS:
                        [List measurable outcomes and timelines]
                        
                        Format as a clear, actionable checklist with specific timelines and criteria.
                        """
                        
                        prevention_plan = st.session_state.ollama_client.generate(prevention_prompt)
                        engine.analysis_results["prevention_plan"] = prevention_plan
                        
                        st.markdown(f'<div class="success-card">{prevention_plan}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Add download button
                        plan_text = f"""# INJURY PREVENTION & REHABILITATION PLAN
                        
                        Athlete: {st.session_state.athlete_name}
                        Sport: {st.session_state.sport}
                        Injury: {engine.injury_details['type']} ({engine.injury_details['severity']})
                        Injury Date: {engine.injury_date}
                        Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        
                        {prevention_plan}
                        
                        ---
                        Generated by Digital Sports Injury Autopsy Engine
                        """
                        
                        st.download_button(
                            label="📥 Download Prevention Plan (TXT)",
                            data=plan_text,
                            file_name=f"prevention_plan_{st.session_state.athlete_name.replace(' ', '_')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.info("💡 Enable 'Generate Prevention Plan' and ensure Ollama is running to create a comprehensive prevention plan.")
        
        with tab4:
            st.markdown('<div class="sub-header">Detailed Metrics Data</div>', 
                       unsafe_allow_html=True)
            
            # Show dataframe
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            with st.expander("📊 Statistical Summary"):
                st.write(df.describe())
            
            # Export options
            st.markdown("### 📤 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"athlete_metrics_{st.session_state.athlete_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                json_str = df.to_json(orient='records', indent=2, date_format='iso')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"athlete_metrics_{st.session_state.athlete_name.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with tab5:
            st.markdown('<div class="sub-header">Comprehensive Report</div>', 
                       unsafe_allow_html=True)
            
            # Generate report
            report = {
                "report_title": "Digital Sports Injury Autopsy Report",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "athlete": {
                    "name": st.session_state.athlete_name,
                    "sport": st.session_state.sport
                },
                "injury_details": engine.injury_details,
                "analysis_parameters": {
                    "model_used": st.session_state.selected_model,
                    "data_points": len(df),
                    "time_period": f"{len(df)} days",
                    "analysis_date": datetime.now().strftime("%Y-%m-%d")
                },
                "key_metrics": {
                    "training_load": {
                        "mean": float(df['training_load'].mean()),
                        "std": float(df['training_load'].std()),
                        "max": float(df['training_load'].max())
                    },
                    "recovery_score": {
                        "mean": float(df['recovery_score'].mean()),
                        "min": float(df['recovery_score'].min())
                    },
                    "acwr": {
                        "mean": float(df['acwr'].mean()),
                        "max": float(df['acwr'].max()),
                        "high_risk_days": int((df['acwr'] > 1.5).sum())
                    },
                    "sleep": {
                        "mean_hours": float(df['sleep_hours'].mean()),
                        "mean_quality": float(df['sleep_quality'].mean())
                    }
                },
                "analysis_summary": {
                    "biomechanical_insights_count": len(engine.analysis_results["biomechanical_insights"]),
                    "root_causes_identified": len(engine.analysis_results["root_causes"]),
                    "prevention_recommendations": len(engine.analysis_results["prevention_recommendations"]),
                    "has_prevention_plan": bool(engine.analysis_results["prevention_plan"])
                },
                "recommendations": [
                    "Maintain ACWR between 0.8-1.3 for optimal load management",
                    "Implement regular recovery assessments (weekly)",
                    "Establish sleep hygiene protocol (7-9 hours/night)",
                    "Conduct biomechanical screening every 3 months",
                    "Use graduated return-to-play protocol post-injury"
                ]
            }
            
            st.json(report, expanded=False)
            
            # Download report
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"injury_autopsy_report_{st.session_state.athlete_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="analysis-card">
        <h2>👋 Welcome to the Digital Sports Injury Autopsy Engine</h2>
        
        <h3>🔍 What is this?</h3>
        A forensic analysis system that performs post-injury analysis for athletes using:
        - <strong>Multi-role expert analysis</strong> (Biomechanist, Physiotherapist, Data Analyst)
        - <strong>Early warning pattern detection</strong>
        - <strong>Counterfactual "what-if" scenario simulation</strong>
        - <strong>Personalized prevention planning</strong>
        
        <h3>🚀 How to Use:</h3>
        1. <strong>Configure</strong> athlete details in the sidebar
        2. <strong>Ensure Ollama is running</strong> for LLM analysis (optional)
        3. <strong>Click "Run Full Analysis"</strong> to begin
        4. <strong>Explore results</strong> across different tabs
        
        <h3>📊 Sample Analysis Includes:</h3>
        • 90 days of synthetic athlete data with injury simulation
        • Interactive visualizations of training load, recovery, sleep, and ACWR
        • Correlation analysis between different metrics
        • Weekly trend analysis
        • Comprehensive reporting
        
        <h3>⚙️ System Requirements:</h3>
        • Python 3.8+ with required packages
        • Ollama (optional, for LLM analysis)
        • 4GB+ RAM recommended
        </div>
        """, unsafe_allow_html=True)
        
        # Installation instructions in expander
        with st.expander("📋 Installation & Setup Instructions", expanded=False):
            st.markdown("""
            ### 1. Install Python Packages
            ```bash
            pip install streamlit pandas numpy plotly requests
            ```
            
            ### 2. Install Ollama (Optional, for LLM features)
            - Download from: https://ollama.com/
            - Start Ollama service:
            ```bash
            ollama serve
            ```
            - Pull required model (in another terminal):
            ```bash
            ollama pull llama3.2
            ```
            
            ### 3. Run the Application
            ```bash
            streamlit run app.py
            ```
            
            ### 4. Deploy to Streamlit Cloud (Optional)
            1. Push code to GitHub
            2. Go to https://share.streamlit.io
            3. Connect repository
            4. Deploy!
            """)
        
        # Quick stats
        st.markdown("### 📈 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Supported Metrics", "10+", "Load, Recovery, Sleep, etc.")
        with col2:
            st.metric("Analysis Roles", "3", "Biomechanist/Physio/Analyst")
        with col3:
            st.metric("Visualizations", "5+", "Interactive charts")
        with col4:
            st.metric("Export Formats", "3", "CSV, JSON, TXT")

if __name__ == "__main__":
    main()