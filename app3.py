import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sports Injury Autopsy Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4B5563;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .analysis-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .sidebar-section {
        margin-bottom: 2rem;
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
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.api_url = f"{self.base_url}/api/generate"
    
    def generate(self, prompt: str, system_prompt: str = None, 
                temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate response from LLM"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "You are a sports medicine expert providing detailed analysis.",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["response"]
            return f"Error: {response.status_code}"
        except:
            return "⚠️ Ollama not available. Using template responses."

class SportsInjuryAutopsy:
    """Main application class"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.athlete_data = {}
        self.injury_data = {}
        self.analyses = {}
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'metrics' not in st.session_state:
            st.session_state.metrics = []
        if 'analysis_requested' not in st.session_state:
            st.session_state.analysis_requested = False
        if 'athlete_data' not in st.session_state:
            st.session_state.athlete_data = {}
        if 'injury_data' not in st.session_state:
            st.session_state.injury_data = {}
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False
        if 'analysis_generated' not in st.session_state:
            st.session_state.analysis_generated = False
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🏥 Digital Sports Injury Autopsy Engine</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
        A comprehensive forensic analysis system for athletic injuries using AI-powered diagnostics
        </div>
        """, unsafe_allow_html=True)
        
        # Quick info boxes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍", "Multi-Role Analysis", "4 Experts")
        with col2:
            st.metric("📊", "Data Patterns", "10+ Metrics")
        with col3:
            st.metric("🛡️", "Prevention Plan", "Custom")
        with col4:
            st.metric("📈", "Visualization", "Interactive")
    
    def render_sidebar(self):
        """Render sidebar for navigation and settings"""
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/2092/2092647.png", width=100)
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            # Ollama connection status
            st.markdown("### Ollama is installed")
            
            # Model selection
            model_option = st.selectbox(
                "Select Model:",
                ["llama3.2", "mistral", "orca-mini", "Template Mode"],
                key="model_select"
            )
            
            # Test connection button
            if st.button("Test Ollama Connection", use_container_width=True):
                try:
                    response = requests.get(f"{self.ollama.base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Ollama is running!")
                    else:
                        st.error("❌ Ollama not responding")
                except:
                    st.error("❌ Cannot connect to Ollama")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("## Analysis Options")
            
            # Days to analyze
            days_option = st.radio(
                "Days to analyze:",
                ["30", "90", "180"],
                key="days_select"
            )
            
            # Include checkboxes
            include_counterfactual = st.checkbox("Include Counterfactual Analysis", value=True, key="counterfactual_check")
            generate_prevention = st.checkbox("Generate Prevention Plan", value=True, key="prevention_check")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Run Analysis button in sidebar
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                if not st.session_state.athlete_data:
                    st.warning("Please fill in athlete information first!")
                else:
                    st.session_state.run_analysis = True
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            <div class='info-box'>
            <small>💡 <b>Tip:</b> Fill in all athlete information on the main page before running analysis.</small>
            </div>
            """, unsafe_allow_html=True)
            
            return model_option, int(days_option), include_counterfactual, generate_prevention
    
    def collect_athlete_info(self):
        """Collect athlete information in an expandable form"""
        with st.expander("👤 Athlete Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                self.athlete_data['name'] = st.text_input("Full Name", "Alex Johnson", key="athlete_name")
                self.athlete_data['age'] = st.number_input("Age", 15, 50, 24, key="athlete_age")
                self.athlete_data['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"], key="athlete_gender")
            with col2:
                self.athlete_data['sport'] = st.text_input("Sport", "Soccer", key="athlete_sport")
                self.athlete_data['position'] = st.text_input("Position/Role", "Midfielder", key="athlete_position")
                self.athlete_data['years_experience'] = st.number_input("Years in Sport", 1, 30, 10, key="athlete_experience")
            
            # Save to session state
            st.session_state.athlete_data = self.athlete_data
    
    def collect_injury_info(self):
        """Collect injury details"""
        with st.expander("🤕 Injury Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                self.injury_data['type'] = st.text_input("Injury Type", "Hamstring Strain", key="injury_type")
                self.injury_data['severity'] = st.selectbox("Severity", ["Grade 1", "Grade 2", "Grade 3", "Severe"], key="injury_severity")
                self.injury_data['date'] = st.date_input("Injury Date", datetime.now(), key="injury_date")
            with col2:
                self.injury_data['mechanism'] = st.selectbox(
                    "Injury Mechanism",
                    ["Sprinting", "Jumping", "Cutting", "Contact", "Overuse", "Non-contact"],
                    key="injury_mechanism"
                )
                self.injury_data['dominant_side'] = st.selectbox("Injured Side", ["Left", "Right", "Bilateral", "Not Applicable"], key="injury_side")
                self.injury_data['previous_same_injury'] = st.checkbox("Previous similar injury?", key="previous_injury_check")
            
            # Save to session state
            st.session_state.injury_data = self.injury_data
    
    def collect_training_data(self):
        """Collect training and recovery data"""
        with st.expander("🏋️ Training & Recovery Metrics", expanded=False):
            
            st.markdown("### Pre-Injury Training (Last 2-4 Weeks)")
            col1, col2 = st.columns(2)
            with col1:
                self.athlete_data['training_intensity'] = st.slider("Training Intensity (1-10)", 1, 10, 7, key="training_intensity")
                self.athlete_data['training_volume'] = st.slider("Training Volume (hours/week)", 5, 30, 15, key="training_volume")
                self.athlete_data['training_variety'] = st.select_slider("Training Variety", ["Very Repetitive", "Some Variety", "Very Varied"], key="training_variety")
            with col2:
                self.athlete_data['sleep_hours'] = st.slider("Avg Sleep (hours/night)", 4.0, 10.0, 6.5, 0.5, key="sleep_hours")
                self.athlete_data['sleep_quality'] = st.slider("Sleep Quality (1-10)", 1, 10, 6, key="sleep_quality")
                self.athlete_data['stress_level'] = st.slider("Stress Level (1-10)", 1, 10, 5, key="stress_level")
            
            st.markdown("### Recovery & Nutrition")
            col1, col2 = st.columns(2)
            with col1:
                self.athlete_data['nutrition_quality'] = st.slider("Nutrition Quality (1-10)", 1, 10, 7, key="nutrition_quality")
                self.athlete_data['hydration'] = st.select_slider("Hydration", ["Poor", "Adequate", "Excellent"], key="hydration")
                self.athlete_data['recovery_activities'] = st.multiselect(
                    "Recovery Activities",
                    ["Ice/Cryo", "Massage", "Physio", "Stretching", "Foam Rolling", "Compression", "None"],
                    default=["Stretching", "Foam Rolling"],
                    key="recovery_activities"
                )
            with col2:
                self.athlete_data['fatigue_level'] = st.slider("Fatigue Level (1-10)", 1, 10, 6, key="fatigue_level")
                self.athlete_data['performance_trend'] = st.select_slider(
                    "Performance Trend",
                    ["Declining", "Stable", "Improving", "Fluctuating"],
                    key="performance_trend"
                )
                self.athlete_data['pain_before_injury'] = st.text_area("Pain/Discomfort Before Injury", "Mild hamstring tightness, level 3/10", key="pain_before")
            
            # Save to session state
            st.session_state.athlete_data.update(self.athlete_data)
    
    def collect_medical_history(self):
        """Collect medical history"""
        with st.expander("📚 Medical & Injury History", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                self.athlete_data['previous_injuries'] = st.text_area(
                    "Previous Injuries",
                    "Left hamstring strain 8 months ago, Right ankle sprain 2 years ago",
                    key="prev_injuries"
                )
                self.athlete_data['surgery_history'] = st.text_area("Surgery History", "None", key="surgery_history")
            with col2:
                self.athlete_data['medications'] = st.text_area("Current Medications", "None", key="medications")
                self.athlete_data['allergies'] = st.text_area("Allergies", "None", key="allergies")
            
            # Save to session state
            st.session_state.athlete_data.update(self.athlete_data)
    
    def collect_biomechanical_observations(self):
        """Collect biomechanical observations"""
        with st.expander("🧬 Biomechanical Observations", expanded=False):
            self.injury_data['biomechanical_notes'] = st.text_area(
                "Observations Before/During Injury",
                "Reduced hip extension during sprints, increased anterior pelvic tilt, decreased glute activation on injured side",
                key="biomech_notes"
            )
            
            st.markdown("### Movement Screening")
            col1, col2 = st.columns(2)
            with col1:
                self.athlete_data['movement_quality'] = st.select_slider(
                    "Overall Movement Quality",
                    ["Poor", "Fair", "Good", "Excellent"],
                    key="movement_quality"
                )
                self.athlete_data['symmetry'] = st.select_slider(
                    "Left-Right Symmetry",
                    ["Very Asymmetric", "Some Asymmetry", "Symmetric"],
                    key="symmetry"
                )
            with col2:
                self.athlete_data['mobility_issues'] = st.multiselect(
                    "Noticed Mobility Issues",
                    ["Hip", "Ankle", "Thoracic", "Shoulder", "None"],
                    default=["Hip"],
                    key="mobility_issues"
                )
                self.athlete_data['strength_imbalances'] = st.multiselect(
                    "Suspected Strength Imbalances",
                    ["Quad/Hamstring", "Glute", "Core", "Upper Body", "None"],
                    default=["Glute"],
                    key="strength_imbalances"
                )
            
            # Save to session state
            st.session_state.injury_data.update(self.injury_data)
            st.session_state.athlete_data.update(self.athlete_data)
    
    def generate_sample_metrics(self, days=90):
        """Generate sample metrics for visualization"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate training data with injury on last 10% of timeline
        metrics = []
        injury_day_index = int(days * 0.8)  # Injury at 80% of timeline
        
        for i, date in enumerate(dates):
            base_load = 60 + 10 * np.sin(i * 0.1) + np.random.normal(0, 3)
            injury_day = (i == injury_day_index)
            
            # Simulate fatigue accumulation
            fatigue_factor = min(1.5, 1.0 + (i / days) * 0.5)
            
            metric = AthleteMetrics(
                date=date.strftime("%Y-%m-%d"),
                training_load=max(30, min(100, base_load * fatigue_factor)),
                sleep_hours=float(max(4, min(10, 7.5 - (base_load - 60) * 0.02 - (i/days) + np.random.normal(0, 0.3)))),
                sleep_quality=float(max(4, min(10, 7 - (base_load - 60) * 0.03 - (i/days)*0.5 + np.random.normal(0, 0.5)))),
                nutrition_score=float(max(4, min(10, 7 - (i/days)*0.3 + np.random.normal(0, 0.5)))),
                stress_level=float(min(10, 4 + (base_load - 60) * 0.05 + (i/days)*2 + np.random.normal(0, 0.5))),
                recovery_score=float(max(4, min(10, 8 - (base_load - 60) * 0.04 - (i/days) + np.random.normal(0, 0.5)))),
                acute_load=float(base_load * 0.95 * fatigue_factor),
                chronic_load=float(base_load * 0.85 * (1 + (i/days)*0.3)),
                acwr=float(min(2.0, 1.1 + (i/days)*0.8 + np.random.normal(0, 0.1))),
                injury_occurred=injury_day,
                injury_type=self.injury_data.get('type', "Unknown Injury") if injury_day else None,
                injury_severity=self.injury_data.get('severity', "Unknown") if injury_day else None,
                biomechanical_notes=self.injury_data.get('biomechanical_notes', "No notes") if injury_day else None
            )
            metrics.append(metric)
        
        return metrics
    
    def visualize_metrics(self, metrics):
        """Create interactive visualizations"""
        df = pd.DataFrame([asdict(m) for m in metrics])
        df['date'] = pd.to_datetime(df['date'])
        
        st.markdown('<div class="section-header">📈 Training & Recovery Trends</div>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Training Load", "Recovery Metrics", "ACWR Analysis"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['training_load'], 
                                     mode='lines+markers', name='Training Load',
                                     line=dict(color='#3B82F6', width=3)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['acute_load'], 
                                     mode='lines', name='Acute Load (7-day)',
                                     line=dict(color='#10B981', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['chronic_load'], 
                                     mode='lines', name='Chronic Load (28-day)',
                                     line=dict(color='#8B5CF6', width=2, dash='dot')))
            
            # Highlight injury day
            injury_rows = df[df['injury_occurred']]
            if not injury_rows.empty:
                injury_date = injury_rows.iloc[0]['date']
                # Convert to datetime for Plotly
                if isinstance(injury_date, pd.Timestamp):
                    injury_date = injury_date.to_pydatetime()
                
                fig.add_vline(x=injury_date, line_dash="dash", line_color="red",
                            annotation_text="Injury", annotation_position="top right")
            
            fig.update_layout(
                title="Training Load Progression",
                xaxis_title="Date",
                yaxis_title="Load (arbitrary units)",
                template="plotly_white",
                height=400,
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            metrics_to_plot = ['sleep_quality', 'recovery_score', 'stress_level', 'nutrition_score']
            colors = ['#10B981', '#3B82F6', '#EF4444', '#F59E0B']
            
            for metric, color in zip(metrics_to_plot, colors):
                fig.add_trace(go.Scatter(x=df['date'], y=df[metric], 
                                         mode='lines+markers', name=metric.replace('_', ' ').title(),
                                         line=dict(color=color, width=2)))
            
            fig.update_layout(
                title="Recovery & Wellness Metrics",
                xaxis_title="Date",
                yaxis_title="Score (1-10)",
                template="plotly_white",
                height=400,
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['acwr'], 
                                     mode='lines+markers', name='ACWR',
                                     line=dict(color='#8B5CF6', width=3)))
            
            # Add safe zone bands
            fig.add_hrect(y0=0.8, y1=1.3, line_width=0, fillcolor="green", opacity=0.1,
                         annotation_text="Optimal Zone", annotation_position="top left")
            fig.add_hrect(y0=1.3, y1=1.5, line_width=0, fillcolor="yellow", opacity=0.1,
                         annotation_text="Caution Zone", annotation_position="top left")
            fig.add_hrect(y0=1.5, y1=3.0, line_width=0, fillcolor="red", opacity=0.1,
                         annotation_text="Danger Zone", annotation_position="top left")
            
            fig.update_layout(
                title="Acute:Chronic Workload Ratio (ACWR)",
                xaxis_title="Date",
                yaxis_title="ACWR",
                template="plotly_white",
                height=400,
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ACWR insights
            st.markdown("### 📊 ACWR Insights")
            latest_acwr = df['acwr'].iloc[-1]
            avg_acwr = df['acwr'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                if latest_acwr < 1.3:
                    st.success(f"✅ Current ACWR: {latest_acwr:.2f} (Optimal Zone)")
                elif latest_acwr < 1.5:
                    st.warning(f"⚠️ Current ACWR: {latest_acwr:.2f} (Caution Zone)")
                else:
                    st.error(f"🚨 Current ACWR: {latest_acwr:.2f} (Danger Zone)")
            
            with col2:
                st.info(f"📈 Average ACWR: {avg_acwr:.2f}")
    
    def generate_biomechanical_analysis(self):
        """Generate biomechanical analysis"""
        st.markdown('<div class="section-header">🧬 Biomechanical Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have athlete data
        if not self.athlete_data:
            st.warning("Please fill in athlete information first.")
            return
        
        prompt = f"""
        ROLE: Sports Biomechanist
        ATHLETE: {self.athlete_data.get('name', 'Unknown')} - {self.athlete_data.get('sport', 'Unknown')} {self.athlete_data.get('position', 'Unknown')}
        INJURY: {self.injury_data.get('type', 'Unknown')} ({self.injury_data.get('severity', 'Unknown')}) during {self.injury_data.get('mechanism', 'Unknown')}
        
        OBSERVATIONS: {self.injury_data.get('biomechanical_notes', 'No observations provided')}
        
        MOVEMENT QUALITY: {self.athlete_data.get('movement_quality', 'Not specified')}
        SYMMETRY: {self.athlete_data.get('symmetry', 'Not specified')}
        MOBILITY ISSUES: {', '.join(self.athlete_data.get('mobility_issues', ['None']))}
        STRENGTH IMBALANCES: {', '.join(self.athlete_data.get('strength_imbalances', ['None']))}
        
        Provide analysis of:
        1. Likely biomechanical failure points
        2. Movement pattern deviations
        3. Muscle activation issues
        4. Sport-specific risk factors
        5. Recommendations for correction
        """
        
        with st.spinner("Generating biomechanical analysis..."):
            analysis = self.ollama.generate(prompt)
        
        st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
        
        # Save analysis
        self.analyses['biomechanical'] = analysis
    
    def generate_physio_analysis(self):
        """Generate physiotherapy analysis"""
        st.markdown('<div class="section-header">🩺 Physiotherapy Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have athlete data
        if not self.athlete_data:
            st.warning("Please fill in athlete information first.")
            return
        
        prompt = f"""
        ROLE: Sports Physiotherapist
        INJURY: {self.injury_data.get('type', 'Unknown')} ({self.injury_data.get('severity', 'Unknown')})
        
        TRAINING BEFORE INJURY:
        - Intensity: {self.athlete_data.get('training_intensity', 'Unknown')}/10
        - Volume: {self.athlete_data.get('training_volume', 'Unknown')} hrs/week
        - Variety: {self.athlete_data.get('training_variety', 'Unknown')}
        
        RECOVERY STATUS:
        - Sleep: {self.athlete_data.get('sleep_hours', 'Unknown')} hrs, Quality: {self.athlete_data.get('sleep_quality', 'Unknown')}/10
        - Stress: {self.athlete_data.get('stress_level', 'Unknown')}/10
        - Fatigue: {self.athlete_data.get('fatigue_level', 'Unknown')}/10
        - Recovery Activities: {', '.join(self.athlete_data.get('recovery_activities', ['None']))}
        
        PRE-INJURY SYMPTOMS: {self.athlete_data.get('pain_before_injury', 'None reported')}
        PREVIOUS INJURIES: {self.athlete_data.get('previous_injuries', 'None reported')}
        
        Provide analysis of:
        1. Tissue overload mechanisms
        2. Recovery deficit contributions
        3. Previous injury implications
        4. Rehabilitation considerations
        5. Load management recommendations
        """
        
        with st.spinner("Generating physiotherapy analysis..."):
            analysis = self.ollama.generate(prompt)
        
        st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
        
        # Save analysis
        self.analyses['physio'] = analysis
    
    def generate_scc_analysis(self):
        """Generate strength & conditioning analysis"""
        st.markdown('<div class="section-header">💪 Strength & Conditioning Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have athlete data
        if not self.athlete_data:
            st.warning("Please fill in athlete information first.")
            return
        
        prompt = f"""
        ROLE: Strength & Conditioning Coach
        SPORT: {self.athlete_data.get('sport', 'Unknown')}
        POSITION: {self.athlete_data.get('position', 'Unknown')}
        EXPERIENCE: {self.athlete_data.get('years_experience', 'Unknown')} years
        
        TRAINING PATTERN:
        - Volume: {self.athlete_data.get('training_volume', 'Unknown')} hrs/week
        - Intensity: {self.athlete_data.get('training_intensity', 'Unknown')}/10
        - Variety: {self.athlete_data.get('training_variety', 'Unknown')}
        - Performance Trend: {self.athlete_data.get('performance_trend', 'Unknown')}
        
        NUTRITION & HYDRATION:
        - Quality: {self.athlete_data.get('nutrition_quality', 'Unknown')}/10
        - Hydration: {self.athlete_data.get('hydration', 'Unknown')}
        
        Provide analysis of:
        1. Training program design issues
        2. Periodization flaws
        3. Strength imbalances
        4. Energy system development
        5. Specific S&C recommendations for return to play
        """
        
        with st.spinner("Generating S&C analysis..."):
            analysis = self.ollama.generate(prompt)
        
        st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
        
        # Save analysis
        self.analyses['scc'] = analysis
    
    def generate_counterfactual_analysis(self):
        """Generate what-if analysis"""
        st.markdown('<div class="section-header">🔄 Counterfactual Analysis</div>', unsafe_allow_html=True)
        
        if not self.athlete_data:
            st.warning("No athlete data available")
            return
        
        prompt = f"""
        ROLE: Sports Scientist
        ATHLETE: {self.athlete_data.get('name', 'Unknown')}
        INJURY: {self.injury_data.get('type', 'Unknown')}
        
        Given the following data:
        - Training load before injury: {self.athlete_data.get('training_intensity', 'Unknown')}/10
        - Sleep: {self.athlete_data.get('sleep_hours', 'Unknown')} hours
        - Stress: {self.athlete_data.get('stress_level', 'Unknown')}/10
        - Recovery activities: {', '.join(self.athlete_data.get('recovery_activities', ['None']))}
        
        Analyze what could have been done differently to prevent this injury:
        1. Training modifications
        2. Recovery enhancements
        3. Monitoring improvements
        4. Early warning signs missed
        5. Specific interventions at different timepoints
        
        Provide actionable insights.
        """
        
        with st.spinner("Generating counterfactual analysis..."):
            analysis = self.ollama.generate(prompt)
        
        st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
        self.analyses['counterfactual'] = analysis
    
    def generate_prevention_plan(self):
        """Generate comprehensive prevention plan"""
        st.markdown('<div class="section-header">🛡️ Prevention & Rehabilitation Plan</div>', unsafe_allow_html=True)
        
        # Check if we have athlete data
        if not self.athlete_data:
            st.warning("Please fill in athlete information first.")
            return
        
        prompt = f"""
        ROLE: Head of Sports Medicine
        ATHLETE: {self.athlete_data.get('name', 'Unknown')}
        INJURY: {self.injury_data.get('type', 'Unknown')} ({self.injury_data.get('severity', 'Unknown')})
        SPORT: {self.athlete_data.get('sport', 'Unknown')}
        
        Create a comprehensive prevention and rehabilitation plan:
        
        1. IMMEDIATE ACTIONS (First 7 days):
        2. REHABILITATION PHASE (Weeks 2-4):
        3. STRENGTHENING PHASE (Weeks 5-8):
        4. RETURN TO SPORT (Weeks 9-12+):
        5. PREVENTION STRATEGIES:
        6. MONITORING PROTOCOLS:
        7. SUCCESS METRICS:
        
        Include specific exercises, timelines, and progression criteria.
        """
        
        with st.spinner("Generating comprehensive prevention plan..."):
            plan = self.ollama.generate(prompt, max_tokens=1500)
        
        st.markdown(f'<div class="analysis-box">{plan}</div>', unsafe_allow_html=True)
        
        # Save plan
        self.analyses['prevention_plan'] = plan
        
        # Add downloadable plan
        st.download_button(
            label="📥 Download Prevention Plan",
            data=plan,
            file_name=f"prevention_plan_{self.athlete_data.get('name', 'athlete').replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    def render_summary_report(self):
        """Render summary report"""
        st.markdown('<div class="section-header">📋 Summary Report</div>', unsafe_allow_html=True)
        
        # Check if we have data
        if not self.athlete_data:
            st.warning("Please fill in athlete information first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Athlete Profile")
            profile_data = {
                "Name": self.athlete_data.get('name', 'Not specified'),
                "Age": self.athlete_data.get('age', 'Not specified'),
                "Sport": self.athlete_data.get('sport', 'Not specified'),
                "Position": self.athlete_data.get('position', 'Not specified'),
                "Experience": f"{self.athlete_data.get('years_experience', 'Not specified')} years"
            }
            for key, value in profile_data.items():
                st.info(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### Injury Summary")
            injury_summary = {
                "Type": self.injury_data.get('type', 'Not specified'),
                "Severity": self.injury_data.get('severity', 'Not specified'),
                "Mechanism": self.injury_data.get('mechanism', 'Not specified'),
                "Date": self.injury_data.get('date', 'Not specified'),
                "Side": self.injury_data.get('dominant_side', 'Not specified')
            }
            for key, value in injury_summary.items():
                st.warning(f"**{key}:** {value}")
        
        # Risk factors summary
        st.markdown("### Identified Risk Factors")
        risk_factors = [
            f"High training intensity ({self.athlete_data.get('training_intensity', 'N/A')}/10)",
            f"Limited training variety ({self.athlete_data.get('training_variety', 'N/A')})",
            f"Suboptimal sleep ({self.athlete_data.get('sleep_hours', 'N/A')} hours)",
            f"Elevated stress ({self.athlete_data.get('stress_level', 'N/A')}/10)",
            f"Fatigue level: {self.athlete_data.get('fatigue_level', 'N/A')}/10"
        ]
        
        for factor in risk_factors:
            st.markdown(f"• {factor}")
        
        # Analysis completion
        st.markdown("### Analysis Completed")
        if self.analyses:
            st.success(f"✅ {len(self.analyses)} analyses generated")
        
        # Generate full report button
        if st.button("📄 Generate Complete Report", type="secondary"):
            self.generate_complete_report()
    
    def generate_complete_report(self):
        """Generate complete report with all analyses"""
        report = f"""
        ============================================
        SPORTS INJURY AUTOPSY REPORT
        ============================================
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ATHLETE INFORMATION:
        ---------------------
        Name: {self.athlete_data.get('name', 'N/A')}
        Age: {self.athlete_data.get('age', 'N/A')}
        Sport: {self.athlete_data.get('sport', 'N/A')}
        Position: {self.athlete_data.get('position', 'N/A')}
        Experience: {self.athlete_data.get('years_experience', 'N/A')} years
        
        INJURY DETAILS:
        ----------------
        Type: {self.injury_data.get('type', 'N/A')}
        Severity: {self.injury_data.get('severity', 'N/A')}
        Date: {self.injury_data.get('date', 'N/A')}
        Mechanism: {self.injury_data.get('mechanism', 'N/A')}
        Side: {self.injury_data.get('dominant_side', 'N/A')}
        
        TRAINING & RECOVERY METRICS:
        -----------------------------
        Training Intensity: {self.athlete_data.get('training_intensity', 'N/A')}/10
        Training Volume: {self.athlete_data.get('training_volume', 'N/A')} hrs/week
        Sleep: {self.athlete_data.get('sleep_hours', 'N/A')} hours, Quality: {self.athlete_data.get('sleep_quality', 'N/A')}/10
        Stress Level: {self.athlete_data.get('stress_level', 'N/A')}/10
        Fatigue Level: {self.athlete_data.get('fatigue_level', 'N/A')}/10
        Performance Trend: {self.athlete_data.get('performance_trend', 'N/A')}
        
        ANALYSES:
        ----------
        """
        
        for analysis_name, analysis_text in self.analyses.items():
            report += f"\n{analysis_name.upper()}:\n{'-'*40}\n{analysis_text}\n"
        
        # Make report downloadable
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name=f"complete_report_{self.athlete_data.get('name', 'athlete').replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    def show_input_forms(self):
        """Show all input forms when not in analysis mode"""
        st.markdown('<div class="section-header">👥 Athlete & Injury Information</div>', unsafe_allow_html=True)
        
        # Information message
        st.info("ℹ️ Fill in all information below, then click 'Run Full Analysis' in the sidebar")
        
        # Collect all information
        self.collect_athlete_info()
        self.collect_injury_info()
        self.collect_training_data()
        self.collect_medical_history()
        self.collect_biomechanical_observations()
    
    def show_analysis_results(self, days_to_analyze, include_counterfactual, generate_prevention):
        """Show all analysis results"""
        
        # Show analysis progress
        st.markdown(f"### 🔍 Analysis in Progress ({days_to_analyze} days of data)")
        
        # Generate sample metrics based on days selected
        if not st.session_state.metrics:
            with st.spinner("Generating training data..."):
                st.session_state.metrics = self.generate_sample_metrics(days_to_analyze)
        
        # Create tabs for different analysis sections
        analysis_tabs = st.tabs([
            "📊 Data Analysis", 
            "🧬 Biomechanical", 
            "🩺 Physiotherapy",
            "💪 S&C",
            "🛡️ Prevention",
            "📋 Summary"
        ])
        
        with analysis_tabs[0]:
            self.visualize_metrics(st.session_state.metrics)
            
            # Metrics summary
            st.markdown("### Key Metrics Summary")
            metrics_df = pd.DataFrame([asdict(m) for m in st.session_state.metrics])
            latest_metrics = metrics_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Training Load", f"{metrics_df['training_load'].mean():.1f}")
            with col2:
                st.metric("Avg Recovery Score", f"{metrics_df['recovery_score'].mean():.1f}/10")
            with col3:
                st.metric("Avg Sleep Hours", f"{metrics_df['sleep_hours'].mean():.1f}")
            with col4:
                acwr_status = "🟢 Optimal" if latest_metrics['acwr'] < 1.3 else "🟡 Caution" if latest_metrics['acwr'] < 1.5 else "🔴 Danger"
                st.metric("Current ACWR", f"{latest_metrics['acwr']:.2f}", acwr_status)
        
        with analysis_tabs[1]:
            self.generate_biomechanical_analysis()
        
        with analysis_tabs[2]:
            self.generate_physio_analysis()
        
        with analysis_tabs[3]:
            self.generate_scc_analysis()
        
        with analysis_tabs[4]:
            if include_counterfactual:
                self.generate_counterfactual_analysis()
            
            if generate_prevention:
                self.generate_prevention_plan()
            else:
                st.info("Prevention plan generation was not selected")
        
        with analysis_tabs[5]:
            self.render_summary_report()
        
        # Reset button
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
                # Clear analysis state
                st.session_state.run_analysis = False
                st.session_state.analysis_generated = False
                st.session_state.metrics = []
                st.rerun()
    
    def run_analysis(self):
        """Main analysis workflow"""
        # Initialize session state
        self.initialize_session_state()
        
        # Load data from session state if available
        if st.session_state.athlete_data:
            self.athlete_data = st.session_state.athlete_data
        if st.session_state.injury_data:
            self.injury_data = st.session_state.injury_data
        
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        model_option, days_to_analyze, include_counterfactual, generate_prevention = self.render_sidebar()
        
        # Set Ollama model if changed
        if model_option != "Template Mode":
            self.ollama.model = model_option
        
        # Main content area based on analysis state
        if not st.session_state.run_analysis:
            self.show_input_forms()
        else:
            self.show_analysis_results(days_to_analyze, include_counterfactual, generate_prevention)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        🏥 Digital Sports Injury Autopsy Engine • For educational purposes • Consult healthcare professionals for medical advice
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Create and run the application
    app = SportsInjuryAutopsy()
    app.run_analysis()

if __name__ == "__main__":
    main()