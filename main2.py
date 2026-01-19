#!/usr/bin/env python3
"""
Interactive Digital Sports Injury Autopsy Engine 🏥📉
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

class InteractiveInjuryAutopsy:
    """Interactive injury analysis system"""
    
    def __init__(self):
        """Initialize interactive system"""
        self.ollama = OllamaClient(model="mistral")
        self.athlete_data = {}
        self.injury_data = {}
        self.metrics_history = []
        
    def collect_user_input(self):
        """Collect injury and athlete information interactively"""
        print("="*60)
        print("🏥 DIGITAL SPORTS INJURY AUTOPSY - INTERACTIVE MODE")
        print("="*60)
        
        # Collect basic athlete information
        print("\n📋 ATHLETE INFORMATION")
        print("-"*40)
        self.athlete_data['name'] = input("Athlete Name: ").strip() or "Alex Johnson"
        self.athlete_data['age'] = input("Age: ").strip() or "24"
        self.athlete_data['sport'] = input("Sport: ").strip() or "Soccer"
        self.athlete_data['position'] = input("Position/Role: ").strip() or "Midfielder"
        self.athlete_data['years_experience'] = input("Years in Sport: ").strip() or "10"
        
        # Collect injury details
        print("\n🤕 INJURY DETAILS")
        print("-"*40)
        self.injury_data['type'] = input("Injury Type (e.g., Hamstring Strain, ACL Tear): ").strip() or "Hamstring Strain"
        self.injury_data['severity'] = input("Injury Severity (e.g., Grade 1, 2, 3): ").strip() or "Grade 2"
        self.injury_data['date'] = input("Injury Date (YYYY-MM-DD): ").strip() or datetime.now().strftime("%Y-%m-%d")
        self.injury_data['mechanism'] = input("Injury Mechanism (e.g., Sprinting, Jumping, Contact): ").strip() or "Sprinting"
        
        # Collect training and recovery information
        print("\n🏋️ TRAINING & RECOVERY INFORMATION")
        print("-"*40)
        self.collect_training_data()
        
        # Collect medical history
        print("\n📚 MEDICAL & INJURY HISTORY")
        print("-"*40)
        self.athlete_data['previous_injuries'] = input("Previous Injuries (list major ones): ").strip() or "Left hamstring strain 8 months ago"
        self.athlete_data['surgery_history'] = input("Surgery History: ").strip() or "None"
        
        # Collect biomechanical observations
        print("\n🧬 BIOMECHANICAL OBSERVATIONS")
        print("-"*40)
        self.injury_data['biomechanical_notes'] = input("Biomechanical observations before/during injury: ").strip() or "Reduced hip extension, increased anterior pelvic tilt"
        
        # Collect recent performance
        print("\n📊 RECENT PERFORMANCE METRICS")
        print("-"*40)
        self.collect_recent_performance()
        
        print("\n✅ Information collected successfully!")
    
    def collect_training_data(self):
        """Collect recent training and recovery data"""
        print("Please describe the training load in the 2-4 weeks BEFORE the injury:")
        self.athlete_data['training_pre_injury'] = input("Training Description: ").strip() or "High intensity, increased sprint volume"
        
        print("\nSleep patterns before injury (hours per night, quality):")
        self.athlete_data['sleep_pre_injury'] = input("Sleep: ").strip() or "6-7 hours, poor quality"
        
        print("\nNutrition and hydration status:")
        self.athlete_data['nutrition_pre_injury'] = input("Nutrition: ").strip() or "Average, occasional missed meals"
        
        print("\nStress levels and life stressors:")
        self.athlete_data['stress_pre_injury'] = input("Stress: ").strip() or "Moderate, upcoming competition"
        
        print("\nRecovery activities (ice, massage, physio, etc.):")
        self.athlete_data['recovery_activities'] = input("Recovery: ").strip() or "Occasional stretching, no structured recovery"
    
    def collect_recent_performance(self):
        """Collect recent performance metrics"""
        print("Performance trend in last 2 weeks (better/worse/stable):")
        self.athlete_data['performance_trend'] = input("Trend: ").strip() or "Gradual decline"
        
        print("\nFatigue levels (1-10 scale, 10=extremely fatigued):")
        self.athlete_data['fatigue_level'] = input("Fatigue: ").strip() or "7"
        
        print("\nPain/discomfort before injury (location, level 1-10):")
        self.athlete_data['pre_injury_pain'] = input("Pain: ").strip() or "Mild hamstring tightness, level 3"
        
        print("\nTraining monotony (varied or repetitive):")
        self.athlete_data['training_variety'] = input("Variety: ").strip() or "Repetitive, same drills daily"
    
    def generate_analysis_prompts(self):
        """Generate analysis based on collected data"""
        analysis_prompts = []
        
        # Biomechanist Analysis
        biomech_prompt = f"""
        ROLE: Sports Biomechanist
        TASK: Analyze biomechanical factors contributing to injury
        
        ATHLETE PROFILE:
        - Name: {self.athlete_data['name']}
        - Age: {self.athlete_data['age']}
        - Sport: {self.athlete_data['sport']}
        - Position: {self.athlete_data['position']}
        
        INJURY DETAILS:
        - Type: {self.injury_data['type']}
        - Severity: {self.injury_data['severity']}
        - Date: {self.injury_data['date']}
        - Mechanism: {self.injury_data['mechanism']}
        
        BIOMECHANICAL OBSERVATIONS:
        {self.injury_data['biomechanical_notes']}
        
        TRAINING CONTEXT:
        {self.athlete_data['training_pre_injury']}
        
        PERFORMANCE TREND: {self.athlete_data['performance_trend']}
        
        Analyze:
        1. Likely biomechanical breakdown points specific to {self.athlete_data['sport']}
        2. Movement pattern deviations that could have contributed
        3. Muscle activation/coordination issues
        4. Sport-specific biomechanical risk factors
        5. Recommendations for biomechanical screening
        
        Provide structured analysis with specific recommendations.
        """
        analysis_prompts.append(("Biomechanist Analysis", biomech_prompt))
        
        # Physiotherapist Analysis
        physio_prompt = f"""
        ROLE: Sports Physiotherapist
        TASK: Analyze tissue loading and recovery factors
        
        ATHLETE HISTORY:
        - Previous Injuries: {self.athlete_data['previous_injuries']}
        - Surgery History: {self.athlete_data['surgery_history']}
        
        RECOVERY STATUS PRE-INJURY:
        - Sleep: {self.athlete_data['sleep_pre_injury']}
        - Nutrition: {self.athlete_data['nutrition_pre_injury']}
        - Stress: {self.athlete_data['stress_pre_injury']}
        - Recovery Activities: {self.athlete_data['recovery_activities']}
        
        PRE-INJURY SYMPTOMS:
        - Pain/Discomfort: {self.athlete_data['pre_injury_pain']}
        - Fatigue Level: {self.athlete_data['fatigue_level']}/10
        
        TRAINING PATTERN:
        - Variety: {self.athlete_data['training_variety']}
        
        Analyze:
        1. Tissue overload mechanisms for {self.injury_data['type']}
        2. Recovery-deficit contributions
        3. Previous injury implications
        4. Load management failures
        5. Acute vs chronic load considerations
        
        Provide clinical reasoning and specific rehabilitation considerations.
        """
        analysis_prompts.append(("Physiotherapist Analysis", physio_prompt))
        
        # Strength & Conditioning Coach Analysis
        scc_prompt = f"""
        ROLE: Strength & Conditioning Coach
        TASK: Analyze training programming and physical preparation
        
        TRAINING LOAD BEFORE INJURY:
        {self.athlete_data['training_pre_injury']}
        
        INJURY MECHANISM: {self.injury_data['mechanism']}
        
        SPORT DEMANDS: {self.athlete_data['sport']} - {self.athlete_data['position']}
        
        EXPERIENCE LEVEL: {self.athlete_data['years_experience']} years
        
        PERFORMANCE TREND: {self.athlete_data['performance_trend']}
        
        Analyze:
        1. Training program design flaws
        2. Periodization issues
        3. Strength imbalances for {self.athlete_data['sport']}
        4. Energy system development vs demands
        5. Movement competency gaps
        
        Provide specific S&C recommendations for return to play and prevention.
        """
        analysis_prompts.append(("Strength & Conditioning Analysis", scc_prompt))
        
        # Sports Psychologist Analysis
        psych_prompt = f"""
        ROLE: Sports Psychologist
        TASK: Analyze psychological and behavioral factors
        
        STRESS LEVELS: {self.athlete_data['stress_pre_injury']}
        
        FATIGUE: {self.athlete_data['fatigue_level']}/10
        
        PERFORMANCE TREND: {self.athlete_data['performance_trend']}
        
        TRAINING MONOTONY: {self.athlete_data['training_variety']}
        
        Analyze:
        1. Psychological fatigue contributions
        2. Stress-injury relationship
        3. Motivation and adherence factors
        4. Behavioral patterns in training/recovery
        5. Coping mechanisms under pressure
        
        Provide psychological strategies for recovery and future prevention.
        """
        analysis_prompts.append(("Sports Psychologist Analysis", psych_prompt))
        
        return analysis_prompts
    
    def perform_counterfactual_analysis(self):
        """Ask user for what-if scenarios"""
        print("\n" + "="*60)
        print("🔮 COUNTERFACTUAL ANALYSIS")
        print("="*60)
        
        scenarios = []
        
        print("\nLet's explore 'what-if' scenarios for injury prevention.")
        print("Consider what changes might have prevented this injury.\n")
        
        while True:
            print("\nEnter a prevention scenario (or type 'done' to finish):")
            scenario = input("What if... ").strip()
            
            if scenario.lower() == 'done':
                break
            
            if scenario:
                scenarios.append(scenario)
                print(f"✓ Added scenario: {scenario}")
        
        return scenarios
    
    def generate_prevention_plan(self, analyses):
        """Generate comprehensive prevention plan"""
        print("\n" + "="*60)
        print("🛡️ GENERATING PREVENTION PLAN")
        print("="*60)
        
        # Combine all analyses for the prevention plan
        combined_analysis = "\n".join([f"{role}:\n{analysis[:500]}..." for role, analysis in analyses])
        
        prevention_prompt = f"""
        ROLE: Head of Sports Medicine
        TASK: Create actionable injury prevention and return-to-play plan
        
        ATHLETE: {self.athlete_data['name']}
        SPORT: {self.athlete_data['sport']}
        INJURY: {self.injury_data['type']} ({self.injury_data['severity']})
        
        MULTIDISCIPLINARY ANALYSIS SUMMARY:
        {combined_analysis}
        
        Create a comprehensive prevention and rehabilitation plan with:
        
        1. IMMEDIATE ACTIONS (Next 7 days):
           - Medical interventions
           - Pain management
           - Initial mobility work
        
        2. SHORT-TERM MODIFICATIONS (2-4 weeks):
           - Training adjustments
           - Recovery protocols
           - Monitoring systems
        
        3. LONG-TERM SYSTEM CHANGES (1-3 months):
           - Program redesign
           - Biomechanical corrections
           - Load management systems
        
        4. RETURN-TO-PLAY PROTOCOL:
           - Phased approach
           - Criteria for progression
           - Sport-specific reconditioning
        
        5. PREVENTION MONITORING:
           - Key metrics to track
           - Warning signs
           - Communication protocols
        
        6. SUCCESS METRICS:
           - Objective measures
           - Timeline expectations
           - Performance benchmarks
        
        Format as an actionable checklist with specific timelines and responsibilities.
        """
        
        print("Generating personalized prevention plan...")
        prevention_plan = self.ollama.generate(prevention_prompt)
        
        return prevention_plan
    
    def run_interactive_analysis(self):
        """Main interactive analysis workflow"""
        print("\n" + "="*60)
        print("🏁 STARTING INTERACTIVE ANALYSIS")
        print("="*60)
        
        # Check Ollama availability
        llm_available = False
        if check_ollama_installation():
            if self.ollama.check_model_available():
                llm_available = True
                print("\n✅ LLM analysis available")
            else:
                print("\n⚠️ No Ollama models found. Using template responses.")
        else:
            print("\n⚠️ Ollama not installed. Using template responses.")
            print("   Install for full analysis: https://ollama.com/")
        
        # Step 1: Collect user input
        self.collect_user_input()
        
        # Step 2: Generate and display analyses
        print("\n" + "="*60)
        print("🔬 PERFORMING MULTIDISCIPLINARY ANALYSIS")
        print("="*60)
        
        analyses = []
        prompts = self.generate_analysis_prompts()
        
        for role, prompt in prompts:
            print(f"\n{'='*40}")
            print(f"{role.upper()}")
            print('='*40)
            
            if llm_available:
                print(f"Generating {role.lower()}...")
                analysis = self.ollama.generate(prompt)
                print(f"\n{analysis}\n")
                analyses.append((role, analysis))
            else:
                # Template response when LLM is not available
                template = self.get_template_response(role)
                print(f"\n{template}\n")
                analyses.append((role, template))
        
        # Step 3: Counterfactual analysis
        counterfactual_scenarios = self.perform_counterfactual_analysis()
        
        if counterfactual_scenarios and llm_available:
            print("\n" + "="*60)
            print("🔄 ANALYZING COUNTERFACTUAL SCENARIOS")
            print("="*60)
            
            for i, scenario in enumerate(counterfactual_scenarios, 1):
                cf_prompt = f"""
                ROLE: Sports Scientist
                TASK: Counterfactual analysis of injury prevention scenario
                
                INJURY: {self.injury_data['type']} in {self.athlete_data['sport']}
                
                SCENARIO: What if {scenario}
                
                ATHLETE CONTEXT:
                - Sport: {self.athlete_data['sport']}
                - Position: {self.athlete_data['position']}
                - Training: {self.athlete_data['training_pre_injury']}
                - Recovery: {self.athlete_data['recovery_activities']}
                
                Analyze:
                1. Likely impact on injury risk
                2. Implementation requirements
                3. Expected timeline for effects
                4. Potential barriers
                5. Monitoring needs
                
                Provide evidence-based reasoning.
                """
                
                print(f"\nScenario {i}: What if {scenario}")
                cf_analysis = self.ollama.generate(cf_prompt)
                print(f"\n{cf_analysis}\n")
        
        # Step 4: Generate prevention plan
        if llm_available:
            prevention_plan = self.generate_prevention_plan(analyses)
        else:
            prevention_plan = self.get_template_prevention_plan()
        
        # Display prevention plan
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE PREVENTION & REHABILITATION PLAN")
        print("="*60)
        print(prevention_plan)
        
        # Step 5: Generate summary report
        self.generate_final_report(analyses, prevention_plan, counterfactual_scenarios)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nSummary report has been generated for {self.athlete_data['name']}")
        print(f"Injury: {self.injury_data['type']} ({self.injury_data['severity']})")
        print(f"Sport: {self.athlete_data['sport']}")
    
    def get_template_response(self, role):
        """Get template response when LLM is not available"""
        templates = {
            "Biomechanist Analysis": f"""
            Based on the information provided for {self.athlete_data['name']}:
            
            LIKELY BIOMECHANICAL FACTORS:
            1. {self.injury_data['mechanism']} mechanism suggests specific movement breakdown
            2. {self.injury_data['biomechanical_notes']} indicates technical flaws
            3. Sport-specific demands of {self.athlete_data['sport']} may have exposed weaknesses
            
            RECOMMENDATIONS:
            - Video analysis of movement patterns
            - 3D motion capture assessment
            - Sport-specific biomechanical screening
            - Corrective exercise program
            """,
            
            "Physiotherapist Analysis": f"""
            CLINICAL ANALYSIS FOR {self.athlete_data['name']}:
            
            TISSUE LOADING FACTORS:
            1. Previous injury history: {self.athlete_data['previous_injuries']}
            2. Recovery status: {self.athlete_data['sleep_pre_injury']}, {self.athlete_data['stress_pre_injury']}
            3. Pre-injury symptoms: {self.athlete_data['pre_injury_pain']}
            
            REHABILITATION CONSIDERATIONS:
            - Phase-based rehabilitation protocol
            - Load management strategy
            - Recovery optimization
            - Return-to-play criteria
            """,
            
            "Strength & Conditioning Analysis": f"""
            TRAINING ANALYSIS FOR {self.athlete_data['name']}:
            
            PROGRAMMING ISSUES:
            1. Training pattern: {self.athlete_data['training_variety']}
            2. Performance trend: {self.athlete_data['performance_trend']}
            3. Sport demands: {self.athlete_data['sport']} ({self.athlete_data['position']})
            
            S&C RECOMMENDATIONS:
            - Periodization review
            - Strength imbalance assessment
            - Energy system evaluation
            - Movement competency training
            """
        }
        
        return templates.get(role, "Analysis template not available.")
    
    def get_template_prevention_plan(self):
        """Get template prevention plan when LLM is not available"""
        return f"""
        PREVENTION AND REHABILITATION PLAN
        For: {self.athlete_data['name']}
        Injury: {self.injury_data['type']} ({self.injury_data['severity']})
        
        1. IMMEDIATE ACTIONS (Week 1):
           - Medical assessment and diagnosis confirmation
           - Pain and inflammation management
           - Initial protected range of motion exercises
        
        2. REHABILITATION PHASE (Weeks 2-4):
           - Progressive loading protocol
           - Neuromuscular re-education
           - Adjacent joint mobility
        
        3. STRENGTHENING PHASE (Weeks 5-8):
           - Sport-specific strength development
           - Energy system reconditioning
           - Technical skill reintroduction
        
        4. RETURN TO SPORT (Weeks 9-12):
           - Graded exposure to sport demands
           - Monitoring of symptoms
           - Performance testing
        
        5. PREVENTION STRATEGIES:
           - Regular biomechanical screening
           - Load monitoring system
           - Recovery optimization
           - Education on warning signs
        """
    
    def generate_final_report(self, analyses, prevention_plan, counterfactual_scenarios):
        """Generate final summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"injury_report_{self.athlete_data['name'].replace(' ', '_')}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DIGITAL SPORTS INJURY AUTOPSY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("ATHLETE INFORMATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.athlete_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nINJURY DETAILS\n")
            f.write("-"*40 + "\n")
            for key, value in self.injury_data.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("MULTIDISCIPLINARY ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            for role, analysis in analyses:
                f.write(f"{role.upper()}\n")
                f.write("-"*40 + "\n")
                f.write(f"{analysis}\n\n")
            
            if counterfactual_scenarios:
                f.write("\n" + "="*60 + "\n")
                f.write("COUNTERFACTUAL ANALYSIS\n")
                f.write("="*60 + "\n\n")
                for i, scenario in enumerate(counterfactual_scenarios, 1):
                    f.write(f"Scenario {i}: What if {scenario}\n\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("PREVENTION & REHABILITATION PLAN\n")
            f.write("="*60 + "\n\n")
            f.write(prevention_plan)
            
            f.write("\n" + "="*60 + "\n")
            f.write("REPORT GENERATED: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("="*60 + "\n")
        
        print(f"\n📄 Complete report saved to: {filename}")

def main():
    """Main execution function"""
    print("""
    🏥 DIGITAL SPORTS INJURY AUTOPSY ENGINE
    =========================================
    
    This interactive system will guide you through a comprehensive
    analysis of a sports injury using multidisciplinary expertise.
    
    You will provide information about:
    1. The athlete and their sport
    2. The injury details
    3. Training and recovery patterns
    4. Medical history
    
    The system will then provide:
    - Biomechanical analysis
    - Physiotherapy assessment
    - Strength & conditioning evaluation
    - Prevention recommendations
    
    Let's begin...
    """)
    
    # Create interactive system
    system = InteractiveInjuryAutopsy()
    
    # Run interactive analysis
    system.run_interactive_analysis()
    
    print("\n" + "="*60)
    print("🎯 KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. Injury prevention requires multidisciplinary approach
    2. Early warning signs are often present before injury
    3. Load management is critical for injury prevention
    4. Recovery is as important as training
    5. Individualized strategies work best
    
    For enhanced analysis:
    1. Install Ollama: https://ollama.com/
    2. Pull a model: 'ollama pull mistral' or 'ollama pull llama3.2'
    3. Run this script again for AI-powered analysis
    """)

if __name__ == "__main__":
    main()