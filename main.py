#!/usr/bin/env python3
"""
Digital Sports Injury Autopsy Engine 🏥📉
A forensic analysis system for athletic injuries using local LLM via Ollama
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Check if ollama is installed
def check_ollama_installation():
    """Check if Ollama is installed and accessible"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from https://ollama.com/")
        return False

# Install required packages if missing
def install_package(package):
    """Install Python package if not available"""
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_packages = ['pandas', 'numpy', 'scikit-learn', 'requests']
for package in required_packages:
    install_package(package)

import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

@dataclass
class AthleteMetrics:
    """Data class for athlete metrics"""
    date: str
    training_load: float
    sleep_hours: float
    sleep_quality: float  # 1-10 scale
    nutrition_score: float  # 1-10 scale
    stress_level: float  # 1-10 scale
    recovery_score: float  # 1-10 scale
    acute_load: float  # 7-day load
    chronic_load: float  # 28-day load
    acwr: float  # Acute:Chronic Workload Ratio
    injury_occurred: bool = False
    injury_type: Optional[str] = None
    injury_severity: Optional[str] = None
    biomechanical_notes: Optional[str] = None
    previous_injuries: Optional[str] = None

class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            model: Ollama model to use (llama3.2, mistral, orca-mini, etc.)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                print(f"Available models: {models}")
                return models
            return []
        except:
            print("⚠️ Could not fetch models list. Using default.")
            return []
    
    def generate(self, prompt: str, system_prompt: str = None, 
                temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Creativity (0.0 to 1.0)
            max_tokens: Maximum response length
            
        Returns:
            LLM response text
        """
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
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def check_model_available(self) -> bool:
        """Check if selected model is available"""
        if self.model in self.available_models:
            return True
        
        print(f"Model '{self.model}' not found. Available models: {self.available_models}")
        if self.available_models:
            self.model = self.available_models[0]
            print(f"Using '{self.model}' instead.")
            return True
        return False

class InjuryAutopsyEngine:
    """Main engine for forensic injury analysis"""
    
    def __init__(self, athlete_name: str = "John Doe", sport: str = "Soccer"):
        """
        Initialize the autopsy engine
        
        Args:
            athlete_name: Name of the athlete
            sport: Sport type
        """
        self.athlete_name = athlete_name
        self.sport = sport
        self.ollama = OllamaClient(model="mistral")  # Using mistral as it's commonly available
        self.metrics_data = []
        self.injury_date = None
        self.injury_details = {}
        
        # Initialize analysis results
        self.analysis_results = {
            "early_warnings": [],
            "root_causes": [],
            "prevention_recommendations": [],
            "risk_factors": {},
            "biomechanical_insights": []
        }
    
    def generate_sample_data(self, days: int = 90):
        """Generate synthetic athlete data for demonstration"""
        print(f"Generating {days} days of sample data...")
        
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Simulate training cycles with peaks and valleys
            cycle_day = i % 21  # 3-week training cycle
            
            if cycle_day < 14:  # Training phase
                training_load = 60 + 10 * np.sin(cycle_day * 0.5) + np.random.normal(0, 5)
            else:  # Recovery phase
                training_load = 40 + 5 * np.sin(cycle_day * 0.3) + np.random.normal(0, 3)
            
            # Simulate injury on a specific day (day 70)
            injury_occurred = (i == 70)
            injury_type = "Hamstring Strain" if injury_occurred else None
            injury_severity = "Grade 2" if injury_occurred else None
            
            # Generate correlated metrics
            sleep_hours = 8.0 - (training_load - 60) * 0.03 + np.random.normal(0, 0.5)
            sleep_quality = max(1, min(10, 8 - (training_load - 60) * 0.05 + np.random.normal(0, 1)))
            nutrition_score = max(1, min(10, 7 - (training_load - 60) * 0.03 + np.random.normal(0, 1)))
            stress_level = min(10, 3 + (training_load - 60) * 0.1 + np.random.normal(0, 1))
            recovery_score = max(1, min(10, 8 - (training_load - 60) * 0.08 + np.random.normal(0, 1)))
            
            # Calculate acute and chronic loads
            acute_window = 7
            chronic_window = 28
            
            # Calculate ACWR (Acute:Chronic Workload Ratio)
            if i >= chronic_window:
                acute_load = training_load  # Simplified - would be average of last 7 days
                chronic_load = training_load * 0.9  # Simplified - would be average of last 28 days
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
        
        print(f"✅ Generated {days} days of data")
        if self.injury_date:
            print(f"📍 Simulated injury on {self.injury_date}: {self.injury_details['type']}")
    
    def detect_anomalies(self, window_days: int = 14):
        """Detect statistical anomalies in the data before injury"""
        if not self.injury_date:
            print("No injury data found")
            return []
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(m) for m in self.metrics_data])
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter data from 14 days before injury
        injury_datetime = pd.to_datetime(self.injury_date)
        start_date = injury_datetime - timedelta(days=window_days)
        mask = (df['date'] >= start_date) & (df['date'] < injury_datetime)
        
        if not df[mask].empty:
            # Features for anomaly detection
            features = ['training_load', 'sleep_hours', 'sleep_quality', 
                       'nutrition_score', 'stress_level', 'recovery_score', 'acwr']
            X = df[mask][features].fillna(df[mask][features].mean())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)
            
            # Get anomaly dates
            anomaly_dates = df[mask]['date'][anomalies == -1]
            
            for date in anomaly_dates:
                day_data = df[df['date'] == date].iloc[0]
                warning = {
                    "date": date.strftime("%Y-%m-%d"),
                    "metrics": {feature: day_data[feature] for feature in features},
                    "description": self._generate_warning_description(day_data)
                }
                self.analysis_results["early_warnings"].append(warning)
            
            print(f"🔍 Detected {len(anomaly_dates)} statistical anomalies in {window_days} days pre-injury")
        
        return self.analysis_results["early_warnings"]
    
    def _generate_warning_description(self, day_data: pd.Series) -> str:
        """Generate human-readable warning description"""
        warnings = []
        
        if day_data['acwr'] > 1.5:
            warnings.append(f"High workload ratio (ACWR: {day_data['acwr']:.2f})")
        if day_data['recovery_score'] < 5:
            warnings.append(f"Poor recovery score ({day_data['recovery_score']:.1f}/10)")
        if day_data['sleep_hours'] < 6:
            warnings.append(f"Inadequate sleep ({day_data['sleep_hours']:.1f} hours)")
        if day_data['stress_level'] > 7:
            warnings.append(f"High stress level ({day_data['stress_level']:.1f}/10)")
        
        return "; ".join(warnings) if warnings else "Multiple metric deviations detected"
    
    def multi_role_analysis(self):
        """Perform analysis using multiple expert roles"""
        if not self.injury_date:
            print("No injury to analyze")
            return
        
        # Get data for analysis
        df = pd.DataFrame([asdict(m) for m in self.metrics_data])
        
        # Biomechanist analysis
        biomech_prompt = f"""
        ROLE: Sports Biomechanist
        TASK: Analyze the biomechanical factors contributing to {self.injury_details['type']}
        
        ATHLETE: {self.athlete_name}
        SPORT: {self.sport}
        INJURY: {self.injury_details['type']} ({self.injury_details['severity']})
        INJURY DATE: {self.injury_date}
        
        PRE-INJURY BIOMECHANICAL OBSERVATIONS: {df[df['date'] == self.injury_date]['biomechanical_notes'].iloc[0] if not df[df['date'] == self.injury_date].empty else 'None recorded'}
        
        Analyze:
        1. Likely biomechanical breakdown points
        2. Movement pattern deviations preceding injury
        3. Muscle activation/coordination issues
        4. Sport-specific biomechanical risk factors
        
        Provide a structured biomechanical analysis.
        """
        
        print("\n🧬 Biomechanist Analysis:")
        biomech_analysis = self.ollama.generate(biomech_prompt)
        print(biomech_analysis[:500] + "...")
        self.analysis_results["biomechanical_insights"].append(biomech_analysis)
        
        # Physiotherapist analysis
        physio_prompt = f"""
        ROLE: Sports Physiotherapist
        TASK: Analyze tissue loading and recovery factors
        
        INJURY HISTORY: {df['previous_injuries'].iloc[-1] if 'previous_injuries' in df.columns and not df.empty else 'No previous injuries'}
        
        TRAINING LOAD TREND (last 28 days): {df['training_load'].tail(28).tolist() if len(df) >= 28 else 'Insufficient data'}
        
        RECOVERY SCORES (last 14 days): {df['recovery_score'].tail(14).tolist() if len(df) >= 14 else 'Insufficient data'}
        
        Analyze:
        1. Tissue overload mechanisms
        2. Recovery-deficit contributions
        3. Previous injury implications
        4. Load management failures
        
        Focus on clinical reasoning but present conclusions clearly.
        """
        
        print("\n🩺 Physiotherapist Analysis:")
        physio_analysis = self.ollama.generate(physio_prompt)
        print(physio_analysis[:500] + "...")
        self.analysis_results["root_causes"].append(physio_analysis)
        
        # Data Analyst analysis
        data_prompt = f"""
        ROLE: Sports Data Analyst
        TASK: Quantitative pattern detection
        
        DATA SUMMARY (last 30 days pre-injury):
        - Avg Training Load: {df['training_load'].tail(30).mean():.1f}
        - Avg ACWR: {df['acwr'].tail(30).mean():.2f}
        - Avg Sleep: {df['sleep_hours'].tail(30).mean():.1f} hours
        - Avg Recovery Score: {df['recovery_score'].tail(30).mean():.1f}/10
        
        TRENDS:
        - Load progression rate: {(df['training_load'].tail(14).mean() - df['training_load'].tail(28).head(14).mean()):.1f}
        - Recovery decline rate: {(df['recovery_score'].tail(14).mean() - df['recovery_score'].tail(28).head(14).mean()):.1f}
        
        Identify:
        1. Statistical outliers in metrics
        2. Correlations between metrics
        3. Predictive patterns
        4. Threshold violations
        
        Present data-driven insights.
        """
        
        print("\n📊 Data Analyst Analysis:")
        data_analysis = self.ollama.generate(data_prompt)
        print(data_analysis[:500] + "...")
        self.analysis_results["risk_factors"]["quantitative"] = data_analysis
    
    def counterfactual_analysis(self, intervention: Dict[str, Any]):
        """
        Perform what-if analysis based on interventions
        
        Args:
            intervention: Dictionary with intervention parameters
                Example: {'load_reduction': 0.12, 'sleep_increase': 1.0}
        """
        print(f"\n🔮 Counterfactual Analysis: {intervention}")
        
        cf_prompt = f"""
        ROLE: Sports Scientist
        TASK: Counterfactual injury prevention analysis
        
        BASE SCENARIO: Athlete suffered {self.injury_details['type']} on {self.injury_date}
        
        INTERVENTION SCENARIO: What if...
        """
        
        if 'load_reduction' in intervention:
            cf_prompt += f"- Weekly training load was reduced by {intervention['load_reduction']*100}%\n"
        if 'sleep_increase' in intervention:
            cf_prompt += f"- Sleep duration increased by {intervention['sleep_increase']} hours\n"
        if 'recovery_improvement' in intervention:
            cf_prompt += f"- Recovery protocols improved by {intervention['recovery_improvement']*100}%\n"
        
        cf_prompt += """
        Analyze:
        1. Likely impact on injury risk
        2. Required implementation changes
        3. Expected timeline for effects
        4. Secondary benefits
        
        Provide evidence-based counterfactual reasoning.
        """
        
        cf_analysis = self.ollama.generate(cf_prompt)
        print(cf_analysis[:500] + "...")
        self.analysis_results["prevention_recommendations"].append({
            "intervention": intervention,
            "analysis": cf_analysis
        })
    
    def generate_prevention_plan(self):
        """Generate comprehensive prevention recommendations"""
        print("\n🛡️ Generating Prevention Plan...")
        
        prevention_prompt = f"""
        ROLE: Head of Sports Medicine
        TASK: Create actionable injury prevention plan
        
        ATHLETE: {self.athlete_name}
        INJURY: {self.injury_details['type']}
        
        ANALYSIS SUMMARY:
        - Biomechanical: {self.analysis_results['biomechanical_insights'][0][:200] if self.analysis_results['biomechanical_insights'] else 'N/A'}
        - Root Causes: {self.analysis_results['root_causes'][0][:200] if self.analysis_results['root_causes'] else 'N/A'}
        - Early Warnings: {len(self.analysis_results['early_warnings'])} anomalies detected
        
        Create a structured prevention plan with:
        1. Immediate actions (next 7 days)
        2. Short-term modifications (2-4 weeks)
        3. Long-term system changes (1-3 months)
        4. Monitoring protocols
        5. Success metrics
        
        Format as actionable checklist.
        """
        
        prevention_plan = self.ollama.generate(prevention_prompt)
        self.analysis_results["prevention_plan"] = prevention_plan
        
        print("="*60)
        print("INJURY PREVENTION PLAN")
        print("="*60)
        print(prevention_plan)
        
        return prevention_plan
    
    def run_full_autopsy(self):
        """Execute complete autopsy pipeline"""
        print("="*60)
        print("DIGITAL SPORTS INJURY AUTOPSY ENGINE")
        print("="*60)
        print(f"Athlete: {self.athlete_name}")
        print(f"Sport: {self.sport}")
        print()
        
        # Check Ollama availability
        if not check_ollama_installation():
            print("⚠️ Running in simulation mode (no LLM analysis)")
            print("   Install Ollama for full analysis: https://ollama.com/")
            simulation_mode = True
        elif not self.ollama.check_model_available():
            print("⚠️ No Ollama models available. Using simulation.")
            simulation_mode = True
        else:
            simulation_mode = False
        
        # Generate sample data
        self.generate_sample_data(days=90)
        
        # Detect anomalies
        print("\n" + "="*60)
        print("PHASE 1: EARLY WARNING DETECTION")
        print("="*60)
        anomalies = self.detect_anomalies()
        
        for warning in anomalies[:3]:  # Show first 3 warnings
            print(f"⚠️ {warning['date']}: {warning['description']}")
        
        # Multi-role analysis (if LLM available)
        if not simulation_mode:
            print("\n" + "="*60)
            print("PHASE 2: MULTI-ROLE FORENSIC ANALYSIS")
            print("="*60)
            self.multi_role_analysis()
            
            print("\n" + "="*60)
            print("PHASE 3: COUNTERFACTUAL ANALYSIS")
            print("="*60)
            self.counterfactual_analysis({'load_reduction': 0.12, 'sleep_increase': 1.0})
            self.counterfactual_analysis({'recovery_improvement': 0.20, 'load_reduction': 0.08})
            
            print("\n" + "="*60)
            print("PHASE 4: PREVENTION STRATEGY")
            print("="*60)
            self.generate_prevention_plan()
        else:
            print("\n" + "="*60)
            print("SIMULATION MODE RESULTS")
            print("="*60)
            print("In a full analysis with Ollama, the system would:")
            print("1. Perform biomechanical analysis of movement patterns")
            print("2. Conduct physiotherapist assessment of tissue loading")
            print("3. Run data analytics on metric correlations")
            print("4. Generate what-if scenarios for prevention")
            print("5. Create personalized recovery protocols")
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate final summary report"""
        print("\n" + "="*60)
        print("AUTOPSY SUMMARY REPORT")
        print("="*60)
        
        report = {
            "athlete": self.athlete_name,
            "sport": self.sport,
            "injury": self.injury_details,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "early_warnings_count": len(self.analysis_results["early_warnings"]),
            "key_risk_factors": [
                "High acute:chronic workload ratio",
                "Poor recovery score trends", 
                "Inadequate sleep duration",
                "Previous injury history"
            ],
            "prevention_focus_areas": [
                "Load management optimization",
                "Sleep and recovery enhancement",
                "Biomechanical screening",
                "Stress monitoring"
            ]
        }
        
        print(json.dumps(report, indent=2))
        
        # Save report to file
        filename = f"injury_autopsy_{self.athlete_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {filename}")

def main():
    """Main execution function"""
    # Create engine instance
    engine = InjuryAutopsyEngine(
        athlete_name="Alex Johnson",
        sport="Basketball"
    )
    
    # Run complete autopsy
    engine.run_full_autopsy()
    
    print("\n" + "="*60)
    print("🏥 AUTOPSY COMPLETE")
    print("="*60)
    print("System has completed forensic analysis of athletic injury.")
    print("\nKey Capabilities Demonstrated:")
    print("✅ Multi-role prompting (Biomechanist/Physio/Analyst)")
    print("✅ Early warning pattern detection")
    print("✅ Counterfactual what-if analysis") 
    print("✅ Chain-of-thought clinical reasoning")
    print("✅ Prevention strategy generation")
    print("\nTo enhance the system:")
    print("1. Install Ollama: https://ollama.com/")
    print("2. Pull a model: 'ollama pull llama3.2'")
    print("3. Connect real athlete data sources")
    print("4. Add sport-specific biomechanical models")

if __name__ == "__main__":
    main()